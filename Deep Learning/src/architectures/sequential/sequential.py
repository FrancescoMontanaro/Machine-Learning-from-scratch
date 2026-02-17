import math
import time
from typing import Optional, Dict, List, Union

from ..base import Architecture
from ...models.data_loader import Split
from ...models import TrainingArguments
from ...core.utils.data_processing import concat
from ...core.utils.context_manager import no_grad
from ...core import Tensor, Module, ModuleList, ModuleOutput


class Sequential(Architecture):
    
    #####################
    ### Magic methods ###
    #####################
    
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

    
    ######################
    ### Public methods ###
    ######################      

    def fit(self, train_args: TrainingArguments) -> Dict[str, list[float]]:
        """
        Method to train the model
        
        Parameters:
        - train_args (TrainingArguments): Training configuration containing data and hyperparameters
        
        Returns:
        - Dict[str, list[float]]: History of the training with losses and metrics
        """
        
        # Initialize training state
        self._init_training_state(train_args)
        
        # Main training loop
        while self.epoch < train_args.num_epochs and not self.stop_training:
            # Shuffle the training data at the beginning of each epoch if shuffle is enabled
            if train_args.shuffle:
                train_args.data_loader.shuffle(Split.TRAIN)
            
            # Training phase
            train_loss_aux = self._run_training_epoch(train_args)
            
            # Validation phase
            val_loss_aux = self._run_validation_epoch(train_args)
            
            # End of epoch
            self._end_epoch(
                train_args = train_args,
                train_loss_aux = train_loss_aux,
                val_loss_aux = val_loss_aux
            )
        
        # Return training history
        return self.history
    

    ##################################
    ### Protected training methods ###
    ##################################
    
    def _init_training_state(self, train_args: TrainingArguments) -> None:
        """
        Initialize training state: history, control variables, lazy init, optimizer
        
        Parameters:
        - train_args (TrainingArguments): Training configuration
        - train_input_names (Iterable[str]): Names of training inputs
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

                # Retrieve a small sample from the data loader for initialization
                train_inputs = train_args.data_loader.train_data.input_tuple
                train_input_names = train_args.data_loader.train_data.input_keys

                # Determine sample size for lazy initialization (use batch size or total samples if smaller)
                sample_size = min(train_args.train_batch_size, train_inputs[0].shape[0])
                sample_inputs = tuple(tensor[0:sample_size] for tensor in train_inputs)
                forward_kwargs = dict(zip(train_input_names, sample_inputs))
                
                # Perform forward pass
                self(**forward_kwargs)
        
        # Set optimizer parameters
        train_args.optimizer.set_parameters(self.parameters)
    
    
    def _run_training_epoch(self, train_args: TrainingArguments) -> Dict[str, float]:
        """
        Run a single training epoch
        
        Parameters:
        - train_args (TrainingArguments): Training configuration
        - n_steps (int): Number of training steps
        
        Returns:
        - Dict[str, float]: Mean auxiliary loss values for the epoch
        """
        
        # Set model to training mode
        self.train()
        
        # Get input names from data loader
        input_names = list(train_args.data_loader.train_data.input.keys())
        
        # Initialize epoch tracking variables
        elapsed_time = 0.0
        epoch_loss = 0.0
        epoch_metrics = {metric.__name__: 0.0 for metric in train_args.metrics}
        epoch_loss_aux: Dict[str, float] = {}

        # Get the number of training steps for progress tracking
        num_steps = train_args.data_loader.num_batches(Split.TRAIN, train_args.train_batch_size)
        
        # Iterate over training batches from the data loader
        for step, (x_batch, y_batch) in enumerate(train_args.data_loader.get_batch(Split.TRAIN, train_args.train_batch_size)):
            # Start timing for the step
            start_time = time.time()
            
            # Forward pass
            batch_kwargs = dict(zip(input_names, x_batch))
            result = self.forward(**batch_kwargs)

            # Compute loss and backpropagate gradients on primary loss tensor only
            loss = self._normalize_loss_output(train_args.loss_fn(y_batch, result.output, **result.aux))
            (loss.output / train_args.gradient_accumulation_steps).backward()
            
            # Update parameters if needed
            if (step + 1) % train_args.gradient_accumulation_steps == 0 or step == num_steps - 1:
                # Update parameters
                train_args.optimizer.update()

                # Zero gradients
                train_args.optimizer.zero_grad()
            
            # Accumulate metrics
            epoch_loss += loss.output.detach().to_numpy().item() 
            self._accumulate_aux_losses(epoch_loss_aux, loss.aux)
            for metric in train_args.metrics:
                epoch_metrics[metric.__name__] += metric(y_batch.detach(), result.output.detach()).detach().to_numpy().item()
            
            # Progress display
            elapsed_time += time.time() - start_time
            ms_per_step = elapsed_time / (step + 1) * 1000
            self._progress_printer.print_progress(
                f"\rEpoch {self.epoch + 1}/{train_args.num_epochs} - Training ({round((step + 1) / num_steps * 100, 2)}%) | "
                f"{self.tensors_in_memory} tensors | {round(ms_per_step, 2)} ms/step --> {self._format_loss_progress(loss, 'loss')}"
            )
        
        # Store epoch results
        self.history["loss"].append(epoch_loss / num_steps)
        for metric in train_args.metrics:
            self.history[metric.__name__].append(epoch_metrics[metric.__name__] / num_steps)
        
        # Return mean auxiliary loss values for epoch-end summary
        return {key: value / num_steps for key, value in epoch_loss_aux.items()}
    
    
    def _run_validation_epoch(self, train_args: TrainingArguments) -> Optional[Dict[str, float]]:
        """
        Run validation for the current epoch
        
        Parameters:
        - train_args (TrainingArguments): Training configuration
        - n_steps (int): Number of validation steps
        
        Returns:
        - Optional[Dict[str, float]]: Mean auxiliary loss values for the validation epoch, or None if no validation data is provided
        """

        # If no validation data is provided, return None
        if train_args.data_loader.valid_data is None:
            return None
        
        # Disable gradient computation during validation
        with no_grad():
            # Set model to evaluation mode
            self.eval()
            
            # Ensure eval_batch_size is set
            assert train_args.eval_batch_size is not None, "eval_batch_size must be set for validation"
            
            # Get input names from data loader
            input_names = list(train_args.data_loader.train_data.input.keys())
            
            # Initialize epoch tracking variables
            elapsed_time = 0.0
            epoch_loss = 0.0
            epoch_metrics = {metric.__name__: 0.0 for metric in train_args.metrics}
            epoch_loss_aux: Dict[str, float] = {}
            
            # Move to a new line for validation progress
            print()

            # Get the number of validation steps for progress tracking
            num_steps = train_args.data_loader.num_batches(Split.VALID, train_args.eval_batch_size)
            
            # Iterate over validation batches from the data loader
            for step, (x_batch, y_batch) in enumerate(train_args.data_loader.get_batch(Split.VALID, train_args.eval_batch_size)):
                # Start timing for the step
                start_time = time.time()
                
                # Forward pass
                batch_kwargs = dict(zip(input_names, x_batch))
                result = self.forward(**batch_kwargs)
                
                # Compute loss for this batch
                batch_loss_out = self._normalize_loss_output(train_args.loss_fn(y_batch, result.output, **result.aux))
                batch_loss = batch_loss_out.output.detach().to_numpy().item()
                
                # Accumulate metrics
                epoch_loss += batch_loss
                self._accumulate_aux_losses(epoch_loss_aux, batch_loss_out.aux)
                for metric in train_args.metrics:
                    epoch_metrics[metric.__name__] += metric(y_batch.detach(), result.output.detach()).detach().to_numpy().item()
                
                # Progress display for validation
                elapsed_time += time.time() - start_time
                ms_per_step = elapsed_time / (step + 1) * 1000
                self._progress_printer.print_progress(
                    f"\r    Epoch {self.epoch + 1}/{train_args.num_epochs} - Validation ({round((step + 1) / num_steps * 100, 2)}%) | "
                    f"{round(ms_per_step, 2)} ms/step --> {self._format_loss_progress(batch_loss_out, 'val_loss')}"
                )
            
            # Store validation results
            self.history["val_loss"].append(epoch_loss / num_steps)
            for metric in train_args.metrics:
                self.history[f"val_{metric.__name__}"].append(epoch_metrics[metric.__name__] / num_steps)
        
        # Return mean auxiliary loss values for epoch-end summary
        return {key: value / num_steps for key, value in epoch_loss_aux.items()}


    def _end_epoch(
        self, 
        train_args: TrainingArguments, 
        train_loss_aux: Optional[Dict[str, float]] = None,
        val_loss_aux: Optional[Dict[str, float]] = None
    ) -> None:
        """
        End of epoch: print progress, execute callbacks, clear cache
        
        Parameters:
        - train_args (TrainingArguments): Training configuration
        - train_loss_aux (Optional[Dict[str, float]]): Mean auxiliary training losses for current epoch
        - val_loss_aux (Optional[Dict[str, float]]): Mean auxiliary validation losses for current epoch
        """

        # Check if validation was performed
        has_validation = train_args.data_loader.valid_data is not None
        
        # If validation was performed, move cursor up one line and clear it
        if has_validation:
            print("\033[A\033[K", end="")
        
        # Build progress message
        msg = f"\rEpoch {self.epoch + 1}/{train_args.num_epochs} --> loss: {self.history['loss'][-1]:.5g}"
        msg += self._format_epoch_aux(train_loss_aux)
        
        # Add metrics to message
        for metric in train_args.metrics:
            msg += f" - {metric.__name__}: {self.history[metric.__name__][-1]:.5g}"
        
        # Add validation results if available
        if has_validation:
            msg += f" | val_loss: {self.history['val_loss'][-1]:.5g}"
            msg += self._format_epoch_aux(val_loss_aux, prefix="val_")
            for metric in train_args.metrics:
                msg += f" - val_{metric.__name__}: {self.history[f'val_{metric.__name__}'][-1]:.5g}"
        
        # Print final epoch results
        self._progress_printer.print_final(msg.ljust(150))
        
        # Increment epoch and execute callbacks
        self.epoch += 1
        for callback in train_args.callbacks:
            callback(self)
        
        # Notify loss function of epoch completion (e.g., for KL annealing in VAE)
        train_args.loss_fn.step_epoch()
        
        # Clear cache
        self.clear_cache()

    
    #########################
    ### Protected helpers ###
    #########################

    def _normalize_loss_output(self, loss_out: Union[Tensor, ModuleOutput]) -> ModuleOutput:
        """
        Normalize loss output to ModuleOutput for backward/progress compatibility.
        
        Parameters:
        - loss_out (Union[Tensor, ModuleOutput]): Loss returned by loss function.
        
        Returns:
        - ModuleOutput: Normalized loss container.
        """
        
        # If the loss function already returns a ModuleOutput, use it directly
        if isinstance(loss_out, ModuleOutput):
            return loss_out
        
        # If the loss function returns a Tensor, wrap it in a ModuleOutput
        if isinstance(loss_out, Tensor):
            return ModuleOutput(output=loss_out)
        
        # If the loss function returns something else, raise an error
        raise TypeError(f"Loss function must return Tensor or ModuleOutput, got {type(loss_out).__name__}")


    def _format_loss_progress(self, loss_out: ModuleOutput, label: str) -> str:
        """
        Build progress string for loss output, including auxiliary tensors if present.
        
        Parameters:
        - loss_out (ModuleOutput): Loss container.
        - label (str): Label for primary loss value (e.g., 'loss', 'val_loss').
        
        Returns:
        - str: Formatted progress snippet.
        """
        
        # Start with primary loss value
        message = f"{label}: {self._format_progress_tensor(loss_out.output)}"
        
        # Append auxiliary loss terms (if any) for progress visibility
        if loss_out.has_aux:
            message += " - " + " - ".join(
                f"{key}: {self._format_progress_tensor(tensor)}"
                for key, tensor in loss_out.aux.items()
            )
        
        # Return the formatted message
        return message
    
    
    def _accumulate_aux_losses(self, accum: Dict[str, float], aux_losses: Dict[str, Tensor]) -> None:
        """
        Accumulate auxiliary loss terms converted to scalars.
        
        Parameters:
        - accum (Dict[str, float]): Running accumulator for auxiliary loss terms.
        - aux_losses (Dict[str, Tensor]): Auxiliary loss tensors from ModuleOutput.
        """
        
        # Convert each auxiliary loss tensor to a scalar and accumulate
        for key, tensor in aux_losses.items():
            accum[key] = accum.get(key, 0.0) + self._progress_scalar_value(tensor)


    def _format_epoch_aux(self, aux_losses: Optional[Dict[str, float]], prefix: str = "") -> str:
        """
        Format mean auxiliary loss terms for epoch-end summary.
        
        Parameters:
        - aux_losses (Optional[Dict[str, float]]): Auxiliary losses averaged over the epoch.
        - prefix (str): Optional key prefix (e.g., 'val_').
        
        Returns:
        - str: Formatted suffix to append to epoch summary message.
        """
        
        # If no auxiliary losses, return empty string
        if not aux_losses:
            return ""
        
        # Format each auxiliary loss term with its prefix and value
        return "".join(
            f" - {prefix}{key}: {value:.5g}"
            for key, value in aux_losses.items()
        )


    @staticmethod
    def _format_progress_tensor(tensor: Tensor) -> str:
        """
        Format tensor value for progress display.
        Scalars are printed directly; non-scalars are summarized by mean.
        """
        
        # Convert tensor to numpy for formatting
        value = tensor.detach().to_numpy()
        
        # If it's a scalar, format directly; if not, format the mean value
        if value.size == 1:
            return f"{value.item():.5g}"
        
        # For non-scalar tensors, return the mean value for progress display
        return f"{value.mean():.5g} (mean)"


    @staticmethod
    def _progress_scalar_value(tensor: Tensor) -> float:
        """
        Convert tensor to scalar for aggregation.
        If non-scalar, the mean value is used.
        """
        
        # Convert tensor to numpy for aggregation
        value = tensor.detach().to_numpy()
        
        # If it's a scalar, return its value; if not, return the mean value for aggregation
        if value.size == 1:
            return float(value.item())
        
        # For non-scalar tensors, return the mean value for aggregation
        return float(value.mean())


    def _forward(
        self, 
        *, 
        batch_size: Optional[int] = None, 
        tensors_to_batch: Optional[list[str]] = None,
        verbose: bool = False, 
        **kwargs
    ) -> Union[Tensor, 'ModuleOutput']:
        """
        Forward pass of the model
        
        Parameters:
        - batch_size (Optional[int]): Number of samples to use for each batch
        - tensors_to_batch (Optional[list[str]]): List of tensor names to batch. If None, all tensors are batched.
        - verbose (bool): Flag to display the progress
        - **kwargs: Named input tensors and additional parameters
        
        Returns:
        - Union[Tensor, ModuleOutput]: Output of the neural network
        
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
        outputs: List[ModuleOutput] = []
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
        
        # After processing all batches, concatenate ModuleOutput results
        # Concatenate the primary tensors
        concatenated_x = concat([r.output for r in outputs], axis=0)
        
        # Concatenate auxiliary tensors (if any)
        concatenated_aux = {}
        if outputs[0].has_aux:
            for key in outputs[0].aux:
                concatenated_aux[key] = concat([r.aux[key] for r in outputs], axis=0)
        
        # Return a ModuleOutput with concatenated results
        return ModuleOutput(output=concatenated_x, **concatenated_aux)
    
    
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