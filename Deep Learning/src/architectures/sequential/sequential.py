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
        X_train: Tensor, 
        y_train: Tensor,
        optimizer: Optimizer,
        loss_fn: LossFn,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        epochs: int = 10,
        metrics: list[Callable[..., Tensor]] = [],
        callbacks: list[Callable] = [],
        shuffle_between_epochs: bool = True,
        X_valid: Optional[Tensor] = None,
        y_valid: Optional[Tensor] = None,
        *args, **kwargs
    ) -> Dict[str, list[Tensor]]:
        """
        Method to train the neural network
        
        Parameters:
        - X_train (Tensor): Features of the training dataset. Shape: (samples, ...)
        - y_train (Tensor): Labels of the training dataset. Shape: (samples, ...)
        - X_valid (Tensor): Features of the validation dataset. Shape: (samples, ...)
        - y_valid (Tensor): Labels of the validation dataset. Shape: (samples, ...)
        - optimizer (Optimizer): Optimizer to update the parameters of the model
        - loss_fn (LossFn): Loss function to compute the error of the model
        - batch_size (int): Number of samples to use for each batch. Default is 8
        - gradient_accumulation_steps (int): Number of steps to accumulate the gradients before updating the parameters. Default is 1
        - epochs (int): Number of epochs to train the model. Default is 10
        - metrics (list[Callable]): List of metrics to evaluate the model. Default is an empty list
        - callbacks (list[Callback]): List of callbacks to execute
        - shuffle_between_epochs (bool): Flag to shuffle the data between epochs. Default is True
        
        Returns:
        - Dict[str, list[Tensor]]: Dictionary containing the history of the model
        
        Raises:
        - ValueError: If the validation set is provided but the validation target is not provided
        """
        
        ############################
        ### Check the input data ###
        ############################
        
        # If the vaidation set is provided check if the validation target is provided
        if X_valid is not None and y_valid is None:
            # Raise an error if the validation target is not provided
            raise ValueError("If the validation set is provided, the validation target must also be provided.")
        
        #######################
        ### Initializations ###
        #######################
        
        # Initialize the history of the model
        self.init_history(metrics)
        
        # Initialize the control variables
        self.epoch, self.stop_training = 0, False
        n_training_steps = max(1, math.ceil(X_train.shape()[0] / batch_size))
        n_valid_steps = max(1, math.ceil(X_valid.shape()[0] / batch_size)) if X_valid is not None else 0
        
        # Execute a first forward pass in evaluation mode to initialize the parameters and their shapes
        with no_grad():
            # Set the model in evaluation mode
            self.eval()
            
            # Compute the output of the model
            self(X_train[:1], *args, **kwargs)
        
        # Set the parameters of the optimizer
        optimizer.set_parameters(self.parameters())
            
        ################################
        ### Start main training loop ###
        ################################
        
        # Iterate over the epochs
        while self.epoch < epochs and not self.stop_training:
            
            ############################
            ### Start training phase ###
            ############################
            
            # Set the model in training mode
            self.train()
            
            # Shuffle the dataset at the beginning of each epoch
            (X_train_shuffled, Y_train_shuffled), _ = shuffle_data((X_train, y_train)) if shuffle_between_epochs else (X_train, y_train)
            
            # Iterate over the batches
            elapsed_time = 0.0
            training_epoch_loss = Tensor(0.0, requires_grad=False)
            train_metrics = {metric.__name__: Tensor(0.0, requires_grad=False) for metric in metrics}
            for training_step in range(n_training_steps):
                # Store the start time
                start_time = time.time()
                
                # Get the current batch of data
                X_training_batch = X_train_shuffled[training_step * batch_size:(training_step + 1) * batch_size]
                y_training_batch = Y_train_shuffled[training_step * batch_size:(training_step + 1) * batch_size]
                
                # Forward pass: Compute the output of the model
                training_batch_output = self.forward(X_training_batch, *args, **kwargs)
                
                # Compute the loss of the model
                training_loss = loss_fn(y_training_batch, training_batch_output)
                    
                # Divide the loss by the number of gradient accumulation steps and execute the backward pass
                (training_loss / gradient_accumulation_steps).backward()
                
                # If the number of accumulation steps is reached or it is the last step, update the parameters
                if (training_step + 1) % gradient_accumulation_steps == 0 or training_step == n_training_steps - 1:
                    # Update the parameters of the model
                    optimizer.update()
                    
                    # Zero the gradients of the parameters
                    optimizer.zero_grad()
                    
                # Update the epoch loss
                training_epoch_loss += training_loss.detach()
                
                # Compute the metrics
                for metric in metrics:
                    train_metrics[metric.__name__] += metric(y_training_batch.detach(), training_batch_output.detach())
                        
                # Comute the the statistics
                end_time = time.time() # Store the end time
                elapsed_time += (end_time - start_time) # Update the elapsed time
                ms_per_step = elapsed_time / (training_step + 1) * 1000 # Compute the milliseconds per step
                tensors_in_memory = self.count_tensors_in_memory() # Compute the number of tensors in memory
                
                # Display epoch progress
                print(f"\rEpoch {self.epoch + 1}/{epochs} ({round((((training_step + 1)/n_training_steps)*100), 2)}%) | {tensors_in_memory} tensors in memory | {round(ms_per_step, 2)} ms/step --> loss: {training_loss.to_numpy():.5g}", end="")
            
            # Store the loss in the history
            self.history["loss"].append(training_epoch_loss / n_training_steps)
            
            # Compute the training metrics
            for metric in metrics:
                # Compute the average of the metrics for the training set and store them
                self.history[metric.__name__].append(train_metrics[metric.__name__] / n_training_steps)
            
            ##############################
            ### Start validation phase ###
            ##############################
            
            # Check if the validation set is provided
            if X_valid is not None and y_valid is not None:
                # Set the model in evaluation mode
                self.eval()
                
                # Disable automatic gradient computation
                with no_grad(): 
                    # Iterate over the validation steps
                    valid_epoch_loss = Tensor(0.0, requires_grad=False)
                    valid_metrics = {metric.__name__: Tensor(0.0, requires_grad=False) for metric in metrics}
                    for valid_step in range(n_valid_steps):
                        # Get the current batch of validation data
                        X_valid_batch = X_valid[valid_step * batch_size:(valid_step + 1) * batch_size]
                        y_valid_batch = y_valid[valid_step * batch_size:(valid_step + 1) * batch_size]
                    
                        # Compute the output of the model for the current validation batch
                        valid_batch_output = self.forward(X_valid_batch, *args, **kwargs)
                        
                        # Compute the loss of the model for the current validation batch
                        # and update the validation epoch loss
                        valid_epoch_loss += loss_fn(y_valid_batch, valid_batch_output).detach()
                        
                        # Compute the metrics
                        for metric in metrics:
                            valid_metrics[metric.__name__] += metric(y_valid_batch, valid_batch_output).detach()
                
                # Store the validation losses in the history
                self.history["val_loss"].append(valid_epoch_loss / n_valid_steps)
            
                # Compute the average metrics for the validation set
                for metric in metrics:
                    # Compute the average of the metrics for the validation set and store them
                    self.history[f"val_{metric.__name__}"].append(valid_metrics[metric.__name__] / n_valid_steps)
        
            #############################
            ### Display the progress  ###
            #############################
            
            # Display progress with metrics
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
                ) if X_valid is not None and y_valid is not None else "".ljust(50)
            )
            
            #############################
            ### Execute the callbacks ###
            #############################
            
            # Increment the epoch counter
            self.epoch += 1
                    
            # Execute the callbacks
            for callback in callbacks:
                # Call the callback
                callback(self)
                
            # Call the garbage collector to free up memory
            self.clear_cache()
         
        # Return the history of the training   
        return self.history
    

    ### Protected methods ###
    
    def _forward(self, x: Tensor, batch_size: Optional[int] = None, verbose: bool = False, *args, **kwargs) -> Tensor:
        """
        Forward pass of the model
        
        Parameters:
        - x (Tensor): Features of the dataset
        - batch_size (Optional[int]): Number of samples to use for each batch
        - verbose (bool): Flag to display the progress
        
        Returns:
        - Tensor: Output of the neural network
        """
        
        # Compute the number of steps to iterate over the batches
        # If the batch size is not provided, set it to 1
        num_steps = max(1, math.ceil(x.shape()[0] / batch_size)) if batch_size else 1
        
        # Initialize the list of outputs
        outputs = []
        
        # Variable to store the time per step
        elapsed_time = 0.0
        
        # Iterate over the batches
        for step in range(num_steps):
            # Store the start time
            start = time.time()
            
            # Get the current batch
            batch_out = x[step * batch_size:(step + 1) * batch_size] if batch_size else x
            
            # Forward pass: Compute the output of the model
            batch_out = self.modules.forward(batch_out, *args, **kwargs)
                
            # Append the output of the batch
            outputs.append(batch_out)
            
            # Store the end time
            end = time.time()
            
            # Update the elapsed time
            elapsed_time += (end - start)
            
            # Display the progress if specified and the number of steps is greater than 1
            # (i.e., there are batches)
            if verbose and num_steps > 1:
                # Compute the time statistics
                ms_per_step = elapsed_time / (step + 1) * 1000

                # Display the progress
                print(f"\rProcessing batch {step + 1}/{num_steps} - {round(ms_per_step, 2)} ms/step", end="")
            
        # Concatenate the outputs
        out = concat(outputs, axis=0)
        
        # Return the output tensor
        return out
    
    
    def _lazy_init(self, *args, **kwargs) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, sequence length, embedding size)
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the number of modules is greater than 0
        assert len(self._modules.values()) > 0, "No modules in the neural network. Add modules to the model!"