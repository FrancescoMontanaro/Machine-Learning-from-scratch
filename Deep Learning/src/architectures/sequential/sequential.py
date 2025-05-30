import math
import time
from typing import Callable, Optional, Dict, Any, Union

from ..base import Architecture
from ...optimizers import Optimizer
from ...loss_functions import LossFn
from ...core import Tensor, Module, ModuleList
from ...core.utils.context_manager import no_grad
from ...core.utils.data_processing import shuffle_data, concat


class Sequential(Architecture):
    
    ### Magic methods ###
    
    def __init__(self, modules: list[Module], *args, **kwargs) -> None:
        """
        Class constructor
        
        Parameters:
        - modules (list[Module]): List of modules to add to the sequential model
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # List of modules of the sequential model
        self.modules: ModuleList = ModuleList(modules)

        
    ### Public methods ###      

    def fit(
        self,
        X_train: Union[Tensor, Dict[str, Tensor]],
        y_train: Tensor,
        optimizer: Optimizer,
        loss_fn: LossFn,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        epochs: int = 10,
        metrics: list[Callable[..., Tensor]] = [],
        callbacks: list[Callable] = [],
        shuffle_between_epochs: bool = True,
        X_valid: Optional[Union[Tensor, Dict[str, Tensor]]] = None,
        y_valid: Optional[Tensor] = None,
        forward_params: Optional[Dict[str, Any]] = {}
    ) -> Dict[str, list[Tensor]]:
        """
        Method to train the model
        
        Parameters:
        - X_train (Union[Tensor, Dict[str, Tensor]]): Single tensor or dictionary of input tensors for training
        - y_train (Tensor): Target tensor for training
        - optimizer (Optimizer): Optimizer to use for training
        - loss_fn (LossFn): Loss function to use for training
        - batch_size (int): Number of samples per batch (default: 8)
        - gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating the parameters (default: 1)
        - epochs (int): Number of epochs to train the model (default: 10)
        - metrics (list[Callable[..., Tensor]]): List of metrics to compute during training (default: [])
        - callbacks (list[Callable]): List of callbacks to execute during training (default: [])
        - shuffle_between_epochs (bool): Whether to shuffle the training data between epochs (default: True)
        - X_valid (Optional[Union[Tensor, Dict[str, Tensor]]]): Single tensor or dictionary of input tensors for validation (default: None)
        - y_valid (Optional[Tensor]): Target tensor for validation (default: None)
        - forward_params (Optional[Dict[str, Any]]): Additional parameters for the forward pass (default: {})
        
        Returns:
        - Dict[str, list[Tensor]]: History of the training with losses and metrics
        """
        
        #############################
        ### Process training data ###
        #############################
        
        # Determine if we're using named inputs (dictionary) or positional inputs (single tensor)
        use_named_inputs = isinstance(X_train, dict)
        
        if isinstance(X_train, Tensor):
            # Single tensor case - use positional arguments
            train_inputs = (X_train,)
            train_input_names = None
            
        elif isinstance(X_train, dict):
            # Dictionary case - use named arguments
            if len(X_train) == 0:
                raise ValueError("Training data dictionary cannot be empty")
            
            # Validate that all values are Tensors
            for key, value in X_train.items():
                if not isinstance(value, Tensor):
                    raise ValueError(f"All values in X_train must be Tensors. Got {type(value)} for key '{key}'")
            
            # Extract tensors and names
            train_input_names = list(X_train.keys())
            train_inputs = tuple(X_train.values())
            
        else:
            # If X_train is neither a Tensor nor a dictionary, raise an error
            raise ValueError("X_train must be a Tensor or a dictionary of Tensors")
        
        # Validate training input consistency
        num_samples_check = self._validate_input_consistency(*train_inputs)
        
        # Validate y_train number of samples
        if y_train.shape()[0] != num_samples_check:
            raise ValueError(f"Target tensor has {y_train.shape()[0]} samples, but input tensors have {num_samples_check} samples")
    
        ###############################
        ### Process validation data ###
        ###############################
        
        # Initialize validation inputs and names
        valid_inputs, valid_input_names, use_named_inputs_valid = None, None, False
        
        if X_valid is not None:
            # Determine validation input type
            use_named_inputs_valid = isinstance(X_valid, dict)
            
            # Validate consistency between training and validation input types
            if use_named_inputs != use_named_inputs_valid:
                raise ValueError(
                    f"Training and validation data must use the same input format. "
                    f"Training uses {'dictionary' if use_named_inputs else 'single tensor'}, "
                    f"validation uses {'dictionary' if use_named_inputs_valid else 'single tensor'}"
                )
            
            if isinstance(X_valid, Tensor):
                # Single tensor case
                valid_inputs = (X_valid,)
                valid_input_names = None
                
            elif isinstance(X_valid, dict):
                # Dictionary case
                if len(X_valid) == 0:
                    raise ValueError("Validation data dictionary cannot be empty")
                
                # Validate that all values are Tensors
                for key, value in X_valid.items():
                    if not isinstance(value, Tensor):
                        raise ValueError(f"All values in X_valid must be Tensors. Got {type(value)} for key '{key}'")
                
                # Extract tensors and names
                valid_input_names = list(X_valid.keys())
                valid_inputs = tuple(X_valid.values())
                
                # Check consistency with training data structure
                if train_input_names is not None:
                    if set(valid_input_names) != set(train_input_names):
                        raise ValueError(f"Validation data input names {valid_input_names} must match training data input names {train_input_names}")
                    # Ensure same order as training data
                    valid_input_names = train_input_names
                    valid_inputs = tuple(X_valid[name] for name in train_input_names)
                    
            else:
                # If X_valid is neither a Tensor nor a dictionary, raise an error
                raise ValueError("X_valid must be a Tensor or a dictionary of Tensors")
            
            # If validation data is provided, then y_valid must also be provided
            if y_valid is None:
                raise ValueError("If X_valid is provided, y_valid must also be provided")
            
            # Validate validation input consistency
            valid_num_samples = self._validate_input_consistency(*valid_inputs)
            
            if y_valid.shape()[0] != valid_num_samples:
                raise ValueError(f"Validation target has {y_valid.shape()[0]} samples, but validation inputs have {valid_num_samples} samples")
        
        # Validate the forward parameters
        if forward_params is None:
            forward_params = {}
        
        #######################
        ### Initializations ###
        #######################
        
        # Initialize the history of the model
        self.init_history(metrics)
        
        # Initialize the control variables
        self.epoch, self.stop_training = 0, False
        n_training_steps = max(1, math.ceil(train_inputs[0].shape()[0] / batch_size))
        n_valid_steps = max(1, math.ceil(valid_inputs[0].shape()[0] / batch_size)) if valid_inputs is not None else 0
        
        # If the module is not initialized, initialize it by executing the lazy initialization through a first forward pass
        if not self.is_initialized():
            # Execute a first forward pass to initialize the module
            with no_grad():
                # Set the model to evaluation mode
                self.eval()
                
                # Slice the inputs to match the batch size
                sample_size = min(batch_size, train_inputs[0].shape()[0])
                sample_inputs = self._slice_inputs(0, sample_size, *train_inputs)
                
                # Create args for forward pass
                forward_args = list(sample_inputs) + list(forward_params.values())
                
                # Execute the lazy initialization
                self(*forward_args)
        
        # Set the parameters of the optimizer
        optimizer.set_parameters(self.parameters())
            
        ################################
        ### Start main training loop ###
        ################################
        
        # Loop until the maximum number of epochs is reached or stop_training is set to True
        while self.epoch < epochs and not self.stop_training:
            
            ############################
            ### Start training phase ###
            ############################
            
            # Set the model to training mode
            self.train()
            
            # Shuffle the dataset at the beginning of each epoch
            if shuffle_between_epochs:
                # Shuffle the training data
                shuffle_tensors = (*train_inputs, y_train)
                shuffled_data, _ = shuffle_data(shuffle_tensors)
                
                # Unpack the shuffled data
                train_inputs_shuffled = shuffled_data[:-1]
                y_train_shuffled = shuffled_data[-1]
                
            else:
                # If not shuffling, keep the original order
                train_inputs_shuffled = train_inputs
                y_train_shuffled = y_train
            
            # Initialize the step variables
            elapsed_time = 0.0
            training_epoch_loss = Tensor(0.0, requires_grad=False)
            train_metrics = {metric.__name__: Tensor(0.0, requires_grad=False) for metric in metrics}
            
            # Iterate over the training steps
            for training_step in range(n_training_steps):
                start_time = time.time()
                
                # Get the current batch of data
                x_training_batch = self._slice_inputs(training_step * batch_size, (training_step + 1) * batch_size, *train_inputs_shuffled)
                y_training_batch = y_train_shuffled[training_step * batch_size:(training_step + 1) * batch_size]
                
                # Forward pass: Choose method based on input type
                if use_named_inputs and train_input_names is not None:
                    # Dictionary case: pass tensors as named arguments
                    batch_input_dict = {name: tensor for name, tensor in zip(train_input_names, x_training_batch)}
                    training_batch_output = self.forward(**batch_input_dict, **forward_params)
                else:
                    # Single tensor case: pass as positional arguments
                    forward_args = list(x_training_batch) + list(forward_params.values())
                    training_batch_output = self.forward(*forward_args)
                
                # Compute the loss of the model
                training_loss = loss_fn(y_training_batch, training_batch_output)
                    
                # Divide the loss by the number of gradient accumulation steps and execute the backward pass
                (training_loss / gradient_accumulation_steps).backward()
                
                # If the number of accumulation steps is reached or it is the last step, update the parameters
                if (training_step + 1) % gradient_accumulation_steps == 0 or training_step == n_training_steps - 1:
                    # Update the parameters of the model
                    optimizer.update()
                    
                    # Reset the gradients of the optimizer
                    optimizer.zero_grad()
                    
                # Update the epoch loss
                training_epoch_loss += training_loss.detach()
                
                # Compute the metrics
                for metric in metrics:
                    train_metrics[metric.__name__] += metric(y_training_batch.detach(), training_batch_output.detach())
                        
                # Compute the statistics
                end_time = time.time()
                elapsed_time += (end_time - start_time)
                ms_per_step = elapsed_time / (training_step + 1) * 1000
                tensors_in_memory = self.count_tensors_in_memory()
                
                # Display epoch progress
                print(f"\rEpoch {self.epoch + 1}/{epochs} ({round((((training_step + 1)/n_training_steps)*100), 2)}%) | {tensors_in_memory} tensors in memory | {round(ms_per_step, 2)} ms/step --> loss: {training_loss.to_numpy():.5g}", end="")
            
            # Store the loss in the history
            self.history["loss"].append(training_epoch_loss / n_training_steps)
            
            # Compute the training metrics
            for metric in metrics:
                self.history[metric.__name__].append(train_metrics[metric.__name__] / n_training_steps)
            
            ##############################
            ### Start validation phase ###
            ##############################
            
            # If validation data is provided, evaluate the model on the validation set
            if valid_inputs is not None and y_valid is not None:
                # Disable gradient computation for validation
                with no_grad(): 
                    # Set the model to evaluation mode
                    self.eval()
                    
                    # Define the validation variables
                    valid_epoch_loss = Tensor(0.0, requires_grad=False)
                    valid_metrics = {metric.__name__: Tensor(0.0, requires_grad=False) for metric in metrics}
                    
                    # Iterate over the validation steps
                    for valid_step in range(n_valid_steps):
                        # Get the current batch of validation data
                        x_valid_batch = self._slice_inputs(valid_step * batch_size, (valid_step + 1) * batch_size, *valid_inputs)
                        y_valid_batch = y_valid[valid_step * batch_size:(valid_step + 1) * batch_size]
                    
                        # Forward pass: Choose method based on input type (same as training)
                        if use_named_inputs_valid and valid_input_names is not None:
                            # Dictionary case: pass tensors as named arguments
                            valid_batch_input_dict = {name: tensor for name, tensor in zip(valid_input_names, x_valid_batch)}
                            valid_batch_output = self.forward(**valid_batch_input_dict, **forward_params)
                        else:
                            # Single tensor case: pass as positional arguments
                            valid_forward_args = list(x_valid_batch) + list(forward_params.values())
                            valid_batch_output = self.forward(*valid_forward_args)
                            
                        # Update the validation loss
                        valid_epoch_loss += loss_fn(y_valid_batch, valid_batch_output).detach()
                        
                        # Compute the metrics for the validation batch
                        for metric in metrics:
                            valid_metrics[metric.__name__] += metric(y_valid_batch, valid_batch_output).detach()
                
                # Store the validation losses in the history
                self.history["val_loss"].append(valid_epoch_loss / n_valid_steps)
            
                # Compute the average metrics for the validation set
                for metric in metrics:
                    self.history[f"val_{metric.__name__}"].append(valid_metrics[metric.__name__] / n_valid_steps)
        
            #############################
            ### Display the progress  ###
            #############################
            
            # Print the progress of the epoch
            print(
                f"\rEpoch {self.epoch + 1}/{epochs} --> "
                f"loss: {self.history['loss'][-1].to_numpy().item():.5g}"
                + "".join(
                    [f" - {metric.__name__.replace('_', ' ')}: {self.history[metric.__name__][-1].to_numpy().item():.5g}" for metric in metrics]
                ) +
                (
                    f" | Valid loss: {self.history['val_loss'][-1].to_numpy().item():.5g}"
                    + "".join(
                        [f" - Valid {metric.__name__.replace('_', ' ')}: {self.history[f'val_{metric.__name__}'][-1].to_numpy().item():.5g}" for metric in metrics]
                    ).ljust(50)   
                ) if valid_inputs is not None and y_valid is not None else "".ljust(50)
            )
            
            #############################
            ### Execute the callbacks ###
            #############################
            
            # Increment the epoch counter
            self.epoch += 1
                    
            # Iterate over the callbacks and execute them
            for callback in callbacks:
                # Execute the callback with the model as the argument
                callback(self)
                
            # Clear the cache after each epoch
            self.clear_cache()
        
        # Return the history of the training
        return self.history
    

    ### Protected methods ###

    def _forward(self, *args, batch_size: Optional[int] = None, verbose: bool = False, **kwargs) -> Tensor:
        """
        Forward pass of the model
        
        Parameters:
        - *args: Positional input tensors and additional parameters (for single tensor input)
        - batch_size (Optional[int]): Number of samples to use for each batch
        - verbose (bool): Flag to display the progress
        - **kwargs: Named input tensors and additional parameters (for dictionary input)
        
        Returns:
        - Tensor: Output of the neural network
        """
        
        # Determine if we're using named inputs (kwargs) or positional inputs (args)
        use_named_inputs = bool(kwargs and any(isinstance(v, Tensor) for v in kwargs.values()))
        
        # If using named inputs, ensure all inputs are tensors
        if use_named_inputs:
            # Named inputs case (dictionary input)
            tensor_inputs = {k: v for k, v in kwargs.items() if isinstance(v, Tensor)}
            
            # Ensure at least one tensor input is provided
            if not tensor_inputs:
                raise ValueError("At least one input tensor must be provided in kwargs")
            
            # Validate tensor consistency
            total_samples = self._validate_input_consistency(*tensor_inputs.values())
            
            # Compute batching
            num_steps = max(1, math.ceil(total_samples / batch_size)) if batch_size else 1
            outputs = []
            elapsed_time = 0.0
            
            # Process in batches
            for step in range(num_steps):
                start = time.time()
                
                if batch_size:
                    # Create batch kwargs - slice tensors, keep others as-is
                    batch_kwargs = {}
                    for key, value in kwargs.items():
                        if isinstance(value, Tensor):
                            batch_kwargs[key] = value[step * batch_size:(step + 1) * batch_size]
                        else:
                            batch_kwargs[key] = value
                else:
                    batch_kwargs = kwargs
                
                # Forward pass: Pass named arguments to modules.forward
                batch_out = self.modules.forward(**batch_kwargs)
                outputs.append(batch_out)
                
                # Measure elapsed time for the batch processing
                end = time.time()
                elapsed_time += (end - start)
                
                # If verbose, print the processing time per step
                if verbose and num_steps > 1:
                    # Calculate milliseconds per step
                    ms_per_step = elapsed_time / (step + 1) * 1000
                    
                    # Print the progress
                    print(f"\rProcessing batch {step + 1}/{num_steps} - {round(ms_per_step, 2)} ms/step", end="")
        
        else:
            # Positional inputs case (single tensor input)
            tensor_inputs = [arg for arg in args if isinstance(arg, Tensor)]
            
            # Ensure at least one tensor input is provided
            if not tensor_inputs:
                raise ValueError("At least one input tensor must be provided")
            
            # Validate tensor consistency
            total_samples = self._validate_input_consistency(*tensor_inputs)
            
            # Compute batching
            num_steps = max(1, math.ceil(total_samples / batch_size)) if batch_size else 1
            outputs = []
            elapsed_time = 0.0
            
            # Process in batches
            for step in range(num_steps):
                start = time.time()
                
                if batch_size:
                    # Slice tensors, keep non-tensors as-is
                    batch_args = []
                    for arg in args:
                        if isinstance(arg, Tensor):
                            batch_args.append(arg[step * batch_size:(step + 1) * batch_size])
                        else:
                            batch_args.append(arg)
                else:
                    batch_args = args
                
                # Forward pass: Pass positional arguments to modules.forward
                batch_out = self.modules.forward(*batch_args)
                outputs.append(batch_out)
                
                # Measure elapsed time for the batch processing
                end = time.time()
                elapsed_time += (end - start)
                
                # If verbose, print the processing time per step
                if verbose and num_steps > 1:
                    # Calculate milliseconds per step
                    ms_per_step = elapsed_time / (step + 1) * 1000
                    
                    # Print the progress
                    print(f"\rProcessing batch {step + 1}/{num_steps} - {round(ms_per_step, 2)} ms/step", end="")
        
        # Concatenate outputs and return the result
        return concat(outputs, axis=0)
    
    
    def _lazy_init(self, *args, **kwargs) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - *args: Variable input arguments
        - **kwargs: Variable keyword arguments
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the number of modules is greater than 0
        assert len(self.modules) > 0, "No modules in the neural network. Add modules to the model!"

    
    def _validate_input_consistency(self, *inputs) -> int:
        """
        Validate that all input tensors have the same batch size
        
        Parameters:
        - *inputs: Variable number of input tensors
        
        Returns:
        - int: The batch size
        
        Raises:
        - ValueError: If batch sizes are inconsistent
        """
        
        # Check if inputs are provided
        if not inputs:
            # If no inputs are provided, raise an error
            raise ValueError("At least one input tensor must be provided")
        
        # Check if all inputs have the same number of samples
        num_samples = inputs[0].shape()[0]
        
        # Iterate over the remaining inputs to check their number of samples
        for i, tensor in enumerate(inputs[1:], 1):
            # Get the batch size of the current tensor
            if tensor.shape()[0] != num_samples:
                # If the batch size is inconsistent, raise an error
                raise ValueError(f"Inconsistent number of samples: input 0 has {num_samples} samples, input {i} has {tensor.shape()[0]} samples")
        
        # Return the number of samples (batch size)
        return num_samples
    
    
    def _slice_inputs(self, start_idx: int, end_idx: int, *inputs) -> tuple:
        """
        Slice all input tensors with the same indices
        
        Parameters:
        - start_idx (int): Start index for slicing
        - end_idx (int): End index for slicing
        - *inputs: Variable number of input tensors
        
        Returns:
        - tuple: Sliced input tensors
        """
        
        # Return a tuple of sliced tensors
        return tuple(tensor[start_idx:end_idx] for tensor in inputs)