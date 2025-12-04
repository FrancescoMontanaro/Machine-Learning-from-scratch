import math
import time
from typing import Optional, Dict, Tuple

from ..base import Architecture
from ..config import TrainingArguments
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
        train_args: TrainingArguments
    ) -> Dict[str, list[float]]:
        """
        Method to train the model
        
        Parameters:
        - train_args (TrainingArguments): Training configuration containing data and hyperparameters
        
        Returns:
        - Dict[str, list[float]]: History of the training with losses and metrics
        """
        
        # Extract data from train_args
        train_data = train_args.train_data
        y_train = train_args.y_train
        valid_data = train_args.valid_data
        y_valid = train_args.y_valid
        
        # Extract and validate input data (validation already done in TrainingArguments)
        train_input_names = list(train_data.keys())
        train_inputs = tuple(train_data.values())
        
        # Validate input consistency
        num_samples = self._validate_input_consistency(*train_inputs)
        if y_train.shape[0] != num_samples:
            raise ValueError(f"Target tensor has {y_train.shape[0]} samples, but input tensors have {num_samples} samples")
        
        # Process validation data
        valid_inputs: Optional[Tuple[Tensor, ...]] = None
        if valid_data is not None:
            valid_inputs = tuple(valid_data[name] for name in train_input_names)
            valid_num_samples = self._validate_input_consistency(*valid_inputs)
            if y_valid is not None and y_valid.shape[0] != valid_num_samples:
                raise ValueError(f"Validation target has {y_valid.shape[0]} samples, but inputs have {valid_num_samples} samples")
        
        # Initialize training state
        self._init_training_state(train_args, train_inputs, train_input_names)
        
        # Compute number of steps
        n_training_steps = max(1, math.ceil(train_inputs[0].shape[0] / train_args.train_batch_size))
        n_valid_steps = max(1, math.ceil(valid_inputs[0].shape[0] / train_args.eval_batch_size)) if valid_inputs else 0
        
        # Main training loop
        while self.epoch < train_args.num_epochs and not self.stop_training:
            # Prepare epoch data (shuffle if needed)
            train_inputs_epoch, y_train_epoch = self._prepare_epoch_data(
                train_inputs, y_train, train_args.shuffle
            )
            
            # Training phase
            self._run_training_epoch(
                train_inputs_epoch, y_train_epoch, train_input_names,
                train_args, n_training_steps
            )
            
            # Validation phase
            if valid_inputs is not None and y_valid is not None:
                self._run_validation_epoch(
                    valid_inputs, y_valid, train_input_names,
                    train_args, n_valid_steps
                )
            
            # End of epoch
            self._end_epoch(train_args, valid_inputs is not None)
        
        # Return training history
        return self.history
    

    ### Private training methods ###
    
    def _init_training_state(
        self,
        train_args: TrainingArguments,
        train_inputs: Tuple[Tensor, ...],
        train_input_names: list[str]
    ) -> None:
        """
        Initialize training state: history, control variables, lazy init, optimizer
        
        Parameters:
        - train_args (TrainingArguments): Training configuration
        - train_inputs (Tuple[Tensor, ...]): Training input tensors
        - train_input_names (list[str]): Names of training inputs
        """
        
        # Initialize history and control variables
        self.init_history(train_args.metrics)
        self.epoch, self.stop_training = 0, False
        
        # Lazy initialization if needed
        if not self.is_initialized:
            # Perform a forward pass with a small batch to initialize parameters
            # Disable gradient computation during this pass
            with no_grad():
                # Set model to evaluation mode
                self.eval()

                # Perform a forward pass with a small batch to initialize parameters
                sample_size = min(train_args.train_batch_size, train_inputs[0].shape[0])
                sample_inputs = self._slice_inputs(0, sample_size, *train_inputs)
                forward_kwargs = dict(zip(train_input_names, sample_inputs))
                
                # Perform forward pass
                self(**forward_kwargs)
        
        # Set optimizer parameters
        train_args.optimizer.set_parameters(self.parameters)
    
    
    def _prepare_epoch_data(
        self,
        train_inputs: Tuple[Tensor, ...],
        y_train: Tensor,
        shuffle: bool
    ) -> Tuple[Tuple[Tensor, ...], Tensor]:
        """
        Prepare data for the current epoch (optionally shuffle)
        
        Parameters:
        - train_inputs (Tuple[Tensor, ...]): Training input tensors
        - y_train (Tensor): Target tensor
        - shuffle (bool): Whether to shuffle the data
        
        Returns:
        - Tuple of shuffled/original inputs and targets
        """
        
        # Shuffle data if required
        if shuffle:
            # Shuffle the data
            shuffled_data, _ = shuffle_data((*train_inputs, y_train))
            
            # Cast since we know shuffle_data returns a tuple when given a tuple
            assert isinstance(shuffled_data, tuple), "shuffle_data should return a tuple"

            # Extract shuffled inputs and targets
            inputs_shuffled: Tuple[Tensor, ...] = shuffled_data[:-1] 
            y_shuffled: Tensor = shuffled_data[-1]

            # Return shuffled inputs and targets
            return inputs_shuffled, y_shuffled
        
        # Return original data if no shuffling
        return train_inputs, y_train
    
    
    def _run_training_epoch(
        self,
        train_inputs: Tuple[Tensor, ...],
        y_train: Tensor,
        input_names: list[str],
        train_args: TrainingArguments,
        n_steps: int
    ) -> None:
        """
        Run a single training epoch
        
        Parameters:
        - train_inputs (Tuple[Tensor, ...]): Training input tensors
        - y_train (Tensor): Target tensor
        - input_names (list[str]): Names of input tensors
        - train_args (TrainingArguments): Training configuration
        - n_steps (int): Number of training steps
        """
        
        # Set model to training mode
        self.train()
        
        # Initialize epoch tracking variables
        elapsed_time = 0.0
        epoch_loss = 0.0
        epoch_metrics = {metric.__name__: 0.0 for metric in train_args.metrics}
        
        # Iterate over training steps
        for step in range(n_steps):
            # Start timing for the step
            start_time = time.time()
            
            # Get batch of training data and targets
            x_batch = self._slice_inputs(
                step * train_args.train_batch_size, 
                (step + 1) * train_args.train_batch_size, 
                *train_inputs
            )
            y_batch = y_train[step * train_args.train_batch_size:(step + 1) * train_args.train_batch_size]
            
            # Forward pass
            batch_kwargs = dict(zip(input_names, x_batch))
            output = self.forward(**batch_kwargs)
            
            # Compute loss and backpropagate gradients
            loss = train_args.loss_fn(y_batch, output)
            (loss / train_args.gradient_accumulation_steps).backward()
            
            # Update parameters if needed
            if (step + 1) % train_args.gradient_accumulation_steps == 0 or step == n_steps - 1:
                # Update parameters
                train_args.optimizer.update()

                # Zero gradients
                train_args.optimizer.zero_grad()
            
            # Accumulate metrics
            epoch_loss += loss.detach().to_numpy().item()
            for metric in train_args.metrics:
                epoch_metrics[metric.__name__] += metric(y_batch.detach(), output.detach()).detach().to_numpy().item()
            
            # Progress display
            elapsed_time += time.time() - start_time
            ms_per_step = elapsed_time / (step + 1) * 1000
            self._progress_printer.print_progress(
                f"\rEpoch {self.epoch + 1}/{train_args.num_epochs} - Training ({round((step + 1) / n_steps * 100, 2)}%) | "
                f"{self.tensors_in_memory} tensors | {round(ms_per_step, 2)} ms/step --> loss: {loss.to_numpy():.5g}"
            )
        
        # Store epoch results
        self.history["loss"].append(epoch_loss / n_steps)
        for metric in train_args.metrics:
            self.history[metric.__name__].append(epoch_metrics[metric.__name__] / n_steps)
    
    
    def _run_validation_epoch(
        self,
        valid_inputs: Tuple[Tensor, ...],
        y_valid: Tensor,
        input_names: list[str],
        train_args: TrainingArguments,
        n_steps: int
    ) -> None:
        """
        Run validation for the current epoch
        
        Parameters:
        - valid_inputs (Tuple[Tensor, ...]): Validation input tensors
        - y_valid (Tensor): Validation target tensor
        - input_names (list[str]): Names of input tensors
        - train_args (TrainingArguments): Training configuration
        - n_steps (int): Number of validation steps
        """
        
        # Disable gradient computation during validation
        with no_grad():
            # Set model to evaluation mode
            self.eval()
            
            # Ensure eval_batch_size is set
            assert train_args.eval_batch_size is not None, "eval_batch_size must be set for validation"
            eval_batch_size = train_args.eval_batch_size
            
            # Initialize epoch tracking variables
            elapsed_time = 0.0
            epoch_loss = 0.0
            epoch_metrics = {metric.__name__: 0.0 for metric in train_args.metrics}
            
            # Move to a new line for validation progress
            print()
            
            # Iterate over validation steps
            for step in range(n_steps):
                # Start timing for the step
                start_time = time.time()
                
                # Get batch of validation data and targets
                x_batch = self._slice_inputs(
                    step * eval_batch_size,
                    (step + 1) * eval_batch_size,
                    *valid_inputs
                )
                y_batch = y_valid[step * eval_batch_size:(step + 1) * eval_batch_size]
                
                # Forward pass
                batch_kwargs = dict(zip(input_names, x_batch))
                output = self.forward(**batch_kwargs)
                
                # Compute loss for this batch
                batch_loss = train_args.loss_fn(y_batch, output).detach().to_numpy().item()
                
                # Accumulate metrics
                epoch_loss += batch_loss
                for metric in train_args.metrics:
                    epoch_metrics[metric.__name__] += metric(y_batch.detach(), output.detach()).detach().to_numpy().item()
                
                # Progress display for validation
                elapsed_time += time.time() - start_time
                ms_per_step = elapsed_time / (step + 1) * 1000
                self._progress_printer.print_progress(
                    f"\r    Epoch {self.epoch + 1}/{train_args.num_epochs} - Validation ({round((step + 1) / n_steps * 100, 2)}%) | "
                    f"{round(ms_per_step, 2)} ms/step --> val_loss: {batch_loss:.5g}"
                )
            
            # Store validation results
            self.history["val_loss"].append(epoch_loss / n_steps)
            for metric in train_args.metrics:
                self.history[f"val_{metric.__name__}"].append(epoch_metrics[metric.__name__] / n_steps)
    
    
    def _end_epoch(
        self, 
        train_args: TrainingArguments, 
        has_validation: bool
    ) -> None:
        """
        End of epoch: print progress, execute callbacks, clear cache
        
        Parameters:
        - train_args (TrainingArguments): Training configuration
        - has_validation (bool): Whether validation was performed
        """
        
        # If validation was performed, move cursor up one line and clear it
        if has_validation:
            print("\033[A\033[K", end="")
        
        # Build progress message
        msg = f"\rEpoch {self.epoch + 1}/{train_args.num_epochs} --> loss: {self.history['loss'][-1]:.5g}"
        
        # Add metrics to message
        for metric in train_args.metrics:
            msg += f" - {metric.__name__.replace('_', ' ')}: {self.history[metric.__name__][-1]:.5g}"
        
        # Add validation results if available
        if has_validation:
            msg += f" | val_loss: {self.history['val_loss'][-1]:.5g}"
            for metric in train_args.metrics:
                msg += f" - val_{metric.__name__.replace('_', ' ')}: {self.history[f'val_{metric.__name__}'][-1]:.5g}"
        
        # Print final epoch results
        self._progress_printer.print_final(msg.ljust(150))
        
        # Increment epoch and execute callbacks
        self.epoch += 1
        for callback in train_args.callbacks:
            callback(self)
        
        # Clear cache
        self.clear_cache()


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