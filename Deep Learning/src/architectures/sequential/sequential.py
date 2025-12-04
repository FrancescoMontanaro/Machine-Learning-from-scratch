import math
import time
from typing import Callable, Optional, Dict

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
        *,
        train_data: Dict[str, Tensor],
        y_train: Tensor,
        optimizer: Optimizer,
        loss_fn: LossFn,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        epochs: int = 10,
        metrics: list[Callable[..., Tensor]] = [],
        callbacks: list[Callable] = [],
        shuffle_between_epochs: bool = True,
        valid_data: Optional[Dict[str, Tensor]] = None,
        y_valid: Optional[Tensor] = None
    ) -> Dict[str, list[float]]:
        """
        Method to train the model
        
        Parameters:
        - train_data (Dict[str, Tensor]): Dictionary of named input tensors for training (e.g., {'x': tensor} or {'encoder_input': t1, 'decoder_input': t2})
        - y_train (Tensor): Target tensor for training
        - optimizer (Optimizer): Optimizer to use for training
        - loss_fn (LossFn): Loss function to use for training
        - batch_size (int): Number of samples per batch (default: 8)
        - gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating the parameters (default: 1)
        - epochs (int): Number of epochs to train the model (default: 10)
        - metrics (list[Callable[..., Tensor]]): List of metrics to compute during training (default: [])
        - callbacks (list[Callable]): List of callbacks to execute during training (default: [])
        - shuffle_between_epochs (bool): Whether to shuffle the training data between epochs (default: True)
        - valid_data (Optional[Dict[str, Tensor]]): Dictionary of named input tensors for validation (default: None)
        - y_valid (Optional[Tensor]): Target tensor for validation (default: None)
        
        Returns:
        - Dict[str, list[float]]: History of the training with losses and metrics
        
        Note:
        - All arguments must be passed as keyword arguments
        - Input tensors are passed as a dictionary where keys are the names expected by the model's forward method
        """
        
        #############################
        ### Process training data ###
        #############################
        
        # Validate train_data
        if not isinstance(train_data, dict) or len(train_data) == 0:
            raise ValueError("train_data must be a non-empty dictionary of Tensors")
        
        # Validate that all values are Tensors
        for key, value in train_data.items():
            if not isinstance(value, Tensor):
                raise ValueError(f"All values in train_data must be Tensors. Got {type(value)} for key '{key}'")
        
        # Extract tensors and names
        train_input_names = list(train_data.keys())
        train_inputs = tuple(train_data.values())
        
        # Validate training input consistency
        num_samples_check = self._validate_input_consistency(*train_inputs)
        
        # Validate y_train number of samples
        if y_train.shape[0] != num_samples_check:
            raise ValueError(f"Target tensor has {y_train.shape[0]} samples, but input tensors have {num_samples_check} samples")
    
        ###############################
        ### Process validation data ###
        ###############################
        
        # Initialize validation inputs
        valid_inputs = None
        
        if valid_data is not None:
            # Validate valid_data
            if not isinstance(valid_data, dict) or len(valid_data) == 0:
                raise ValueError("valid_data must be a non-empty dictionary of Tensors")
            
            # Validate that all values are Tensors
            for key, value in valid_data.items():
                if not isinstance(value, Tensor):
                    raise ValueError(f"All values in valid_data must be Tensors. Got {type(value)} for key '{key}'")
            
            # Check consistency with training data structure
            if set(valid_data.keys()) != set(train_input_names):
                raise ValueError(f"Validation data input names {list(valid_data.keys())} must match training data input names {train_input_names}")
            
            # Ensure same order as training data
            valid_inputs = tuple(valid_data[name] for name in train_input_names)
            
            # If validation data is provided, then y_valid must also be provided
            if y_valid is None:
                raise ValueError("If valid_data is provided, y_valid must also be provided")
            
            # Validate validation input consistency
            valid_num_samples = self._validate_input_consistency(*valid_inputs)
            
            if y_valid.shape[0] != valid_num_samples:
                raise ValueError(f"Validation target has {y_valid.shape[0]} samples, but validation inputs have {valid_num_samples} samples")
        
        #######################
        ### Initializations ###
        #######################
        
        # Initialize the history of the model
        self.init_history(metrics)
        
        # Initialize the control variables
        self.epoch, self.stop_training = 0, False
        n_training_steps = max(1, math.ceil(train_inputs[0].shape[0] / batch_size))
        n_valid_steps = max(1, math.ceil(valid_inputs[0].shape[0] / batch_size)) if valid_inputs is not None else 0
        
        # If the module is not initialized, initialize it by executing the lazy initialization through a first forward pass
        if not self.is_initialized:
            # Execute a first forward pass to initialize the module
            with no_grad():
                # Set the model to evaluation mode
                self.eval()
                
                # Slice the inputs to match the batch size
                sample_size = min(batch_size, train_inputs[0].shape[0])
                sample_inputs = self._slice_inputs(0, sample_size, *train_inputs)
                
                # Create kwargs for forward pass using named arguments
                forward_kwargs = {name: tensor for name, tensor in zip(train_input_names, sample_inputs)}
                
                # Execute the lazy initialization
                self(**forward_kwargs)
        
        # Set the parameters of the optimizer
        optimizer.set_parameters(self.parameters)
            
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
            training_epoch_loss = 0.0
            train_metrics = {metric.__name__: 0.0 for metric in metrics}
            
            # Iterate over the training steps
            for training_step in range(n_training_steps):
                start_time = time.time()
                
                # Get the current batch of data
                x_training_batch = self._slice_inputs(training_step * batch_size, (training_step + 1) * batch_size, *train_inputs_shuffled)
                y_training_batch = y_train_shuffled[training_step * batch_size:(training_step + 1) * batch_size]
                
                # Build kwargs for forward pass
                batch_input_dict = {name: tensor for name, tensor in zip(train_input_names, x_training_batch)}
                
                # Execute forward pass with named arguments only
                training_batch_output = self.forward(**batch_input_dict)
                
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
                training_epoch_loss += training_loss.detach().to_numpy().item()
                
                # Compute the metrics
                for metric in metrics:
                    train_metrics[metric.__name__] += metric(y_training_batch.detach(), training_batch_output.detach()).detach().to_numpy().item()
                        
                # Compute the statistics
                end_time = time.time()
                elapsed_time += (end_time - start_time)
                ms_per_step = elapsed_time / (training_step + 1) * 1000
                
                # Display epoch progress
                self._progress_printer.print_progress(f"\rEpoch {self.epoch + 1}/{epochs} ({round((((training_step + 1)/n_training_steps)*100), 2)}%) | {self.tensors_in_memory} tensors in memory | {round(ms_per_step, 2)} ms/step --> loss: {training_loss.to_numpy():.5g}")
            
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
                    valid_epoch_loss = 0.0
                    valid_metrics = {metric.__name__: 0.0 for metric in metrics}
                    
                    # Iterate over the validation steps
                    for valid_step in range(n_valid_steps):
                        # Get the current batch of validation data
                        x_valid_batch = self._slice_inputs(valid_step * batch_size, (valid_step + 1) * batch_size, *valid_inputs)
                        y_valid_batch = y_valid[valid_step * batch_size:(valid_step + 1) * batch_size]
                    
                        # Build kwargs for forward pass (use train_input_names since they are validated to be the same)
                        valid_batch_input_dict = {name: tensor for name, tensor in zip(train_input_names, x_valid_batch)}
                        
                        # Execute forward pass with named arguments only
                        valid_batch_output = self.forward(**valid_batch_input_dict)
                            
                        # Update the validation loss
                        valid_epoch_loss += loss_fn(y_valid_batch, valid_batch_output).detach().to_numpy().item()
                        
                        # Compute the metrics for the validation batch
                        for metric in metrics:
                            valid_metrics[metric.__name__] += metric(y_valid_batch.detach(), valid_batch_output.detach()).detach().to_numpy().item()
                
                # Store the validation losses in the history
                self.history["val_loss"].append(valid_epoch_loss / n_valid_steps)
            
                # Compute the average metrics for the validation set
                for metric in metrics:
                    self.history[f"val_{metric.__name__}"].append(valid_metrics[metric.__name__] / n_valid_steps)
        
            #############################
            ### Display the progress  ###
            #############################
            
            # Print the progress of the epoch
            self._progress_printer.print_final(
                f"\rEpoch {self.epoch + 1}/{epochs} --> "
                f"loss: {self.history['loss'][-1]:.5g}"
                + "".join(
                    [f" - {metric.__name__.replace('_', ' ')}: {self.history[metric.__name__][-1]:.5g}" for metric in metrics]
                ) +
                (
                    f" | Valid loss: {self.history['val_loss'][-1]:.5g}"
                    + "".join(
                        [f" - Valid {metric.__name__.replace('_', ' ')}: {self.history[f'val_{metric.__name__}'][-1]:.5g}" for metric in metrics]
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

    def _forward(
        self, 
        *, 
        batch_size: Optional[int] = None, 
        tensors_to_batch: Optional[list[str]] = None,
        verbose: bool = False, 
        **kwargs
    ) -> Tensor:
        """
        Forward pass of the model
        
        Parameters:
        - batch_size (Optional[int]): Number of samples to use for each batch
        - tensors_to_batch (Optional[list[str]]): List of tensor names to batch. If None, all tensors are batched.
        - verbose (bool): Flag to display the progress
        - **kwargs: Named input tensors and additional parameters
        
        Returns:
        - Tensor: Output of the neural network
        
        Note:
        - All arguments must be passed as keyword arguments (no positional arguments allowed)
        - Only tensors whose names are in tensors_to_batch will be sliced during batching
        """
        
        # Extract tensor inputs from kwargs
        tensor_inputs = {k: v for k, v in kwargs.items() if isinstance(v, Tensor)}
        
        # Ensure at least one tensor input is provided
        if not tensor_inputs:
            raise ValueError("At least one input tensor must be provided in kwargs")
        
        # Determine which tensors should be batched
        if tensors_to_batch is not None:
            # Validate that all specified tensor names exist in kwargs
            for name in tensors_to_batch:
                if name not in tensor_inputs:
                    raise ValueError(f"Tensor '{name}' specified in tensors_to_batch not found in inputs. Available tensors: {list(tensor_inputs.keys())}")
            
            # Get tensors to batch for validation
            tensors_for_batching = {k: v for k, v in tensor_inputs.items() if k in tensors_to_batch}
        else:
            # If no tensors_to_batch specified, batch all tensors
            tensors_for_batching = tensor_inputs
        
        # Validate tensor consistency (only for tensors that will be batched)
        if tensors_for_batching:
            total_samples = self._validate_input_consistency(*tensors_for_batching.values())
        else:
            # If no tensors to batch, no batching is needed
            total_samples = 1
            batch_size = None
        
        # Compute batching
        num_steps = max(1, math.ceil(total_samples / batch_size)) if batch_size else 1
        outputs = []
        elapsed_time = 0.0
        
        # Process in batches
        for step in range(num_steps):
            # Start timing for the batch processing
            start = time.time()
            
            # Build batch kwargs
            if batch_size:
                # Create batch kwargs - slice only tensors in tensors_to_batch
                batch_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, Tensor):
                        # Only slice if tensor is in tensors_to_batch (or if tensors_to_batch is None, slice all)
                        if tensors_to_batch is None or key in tensors_to_batch:
                            batch_kwargs[key] = value[step * batch_size:(step + 1) * batch_size]
                        else:
                            # Tensor not in tensors_to_batch, pass as-is
                            batch_kwargs[key] = value
                    else:
                        # Non-tensor values are passed as-is
                        batch_kwargs[key] = value
            else:
                # No batching, pass all kwargs as-is
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
        num_samples = inputs[0].shape[0]
        
        # Iterate over the remaining inputs to check their number of samples
        for i, tensor in enumerate(inputs[1:], 1):
            # Get the batch size of the current tensor
            if tensor.shape[0] != num_samples:
                # If the batch size is inconsistent, raise an error
                raise ValueError(f"Inconsistent number of samples: input 0 has {num_samples} samples, input {i} has {tensor.shape[0]} samples")
        
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