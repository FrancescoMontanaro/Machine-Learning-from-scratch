import math
import time
import numpy as np
from typing import Callable, Optional

from ..base import Architecture
from ...optimizers import Optimizer
from ...loss_functions import LossFn
from ...core.utils.data_processing import *
from ...core import Tensor, Module, ModuleList
from ...core.utils.context_manager import no_grad


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
        X_valid: Tensor,
        y_valid: Tensor,
        optimizer: Optimizer,
        loss_fn: LossFn,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        epochs: int = 10,
        metrics: list[Callable[..., Tensor]] = [],
        callbacks: list[Callable] = []
    ) -> dict[str, Tensor]:
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
        
        Returns:
        - dict[str, Tensor]: Dictionary containing the training and validation losses
        """
        
        #######################
        ### Initializations ###
        #######################
        
        # Initialize the history of the model
        self.init_history(metrics)
        
        # Initialize the control variables
        self.epoch, self.stop_training = 0, False
        n_training_steps = max(1, math.ceil(X_train.shape()[0] / batch_size))
        n_valid_steps = max(1, math.ceil(X_valid.shape()[0] / batch_size))
        
        # Execute a first forward pass in evaluation mode to initialize the parameters and their shapes
        with no_grad():
            # Set the model in evaluation mode
            self.eval()
            
            # Compute the output of the model
            self(X_train[:1])
        
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
            X_train_shuffled, Y_train_shuffled = shuffle_data((X_train, y_train))
            
            # Iterate over the batches
            elapsed_time = 0.0
            training_epoch_loss = 0.0
            train_metrics = {metric.__name__: 0.0 for metric in metrics}
            for training_step in range(n_training_steps):
                # Store the start time
                start_time = time.time()
                
                # Get the current batch of data
                X_training_batch = X_train_shuffled[training_step * batch_size:(training_step + 1) * batch_size]
                y_training_batch = Y_train_shuffled[training_step * batch_size:(training_step + 1) * batch_size]
                
                # Forward pass: Compute the output of the model
                training_batch_output = self.forward(X_training_batch)
                
                # Compute the loss of the model
                training_loss = loss_fn(y_training_batch, training_batch_output)
                
                # Check if the number of accumulation steps is greater than 1
                if gradient_accumulation_steps > 1:
                    # Scale the loss by the number of accumulation steps
                    training_loss /= gradient_accumulation_steps
                    
                # Execute the backward pass
                training_loss.backward()
                
                # If the number of accumulation steps is reached or it is the last step, update the parameters
                if training_step % gradient_accumulation_steps == 0 or training_step == n_training_steps - 1:
                    # Update the parameters of the model
                    optimizer.update()
                    
                    # Zero the gradients of the parameters
                    optimizer.zero_grad()
                
                # Update the epoch loss
                training_epoch_loss += training_loss.detach().to_numpy().item()
                
                # Compute the metrics
                for metric in metrics:
                    train_metrics[metric.__name__] += metric(y_training_batch, training_batch_output).detach().to_numpy().item()
                    
                # Comute the the statistics
                end_time = time.time() # Store the end time
                elapsed_time += (end_time - start_time) # Update the elapsed time
                ms_per_step = elapsed_time / (training_step + 1) * 1000 # Compute the milliseconds per step
                tensors_in_memory = self.count_tensors_in_memory() # Compute the number of tensors in memory
                
                # Display epoch progress
                print(f"\rEpoch {self.epoch + 1}/{epochs} ({round((((training_step + 1)/n_training_steps)*100), 2)}%) | {tensors_in_memory} tensors in memory | {round(ms_per_step, 2)} ms/step --> loss: {training_loss.to_numpy():.4f}", end="")
            
            ##############################
            ### Start validation phase ###
            ##############################
            
            # Set the model in evaluation mode
            self.eval()
            
            # Disable automatic gradient computation
            with no_grad(): 
                # Iterate over the validation steps
                valid_epoch_loss = 0.0
                valid_metrics = {metric.__name__: 0.0 for metric in metrics}
                for valid_step in range(n_valid_steps):
                    # Get the current batch of validation data
                    X_valid_batch = X_valid[valid_step * batch_size:(valid_step + 1) * batch_size]
                    y_valid_batch = y_valid[valid_step * batch_size:(valid_step + 1) * batch_size]
                
                    # Compute the output of the model for the current validation batch
                    valid_batch_output = self.forward(X_valid_batch)
                    
                    # Compute the loss of the model for the current validation batch
                    # and update the validation epoch loss
                    valid_epoch_loss += loss_fn(y_valid_batch, valid_batch_output).detach().to_numpy().item()
                    
                    # Compute the metrics
                    for metric in metrics:
                        valid_metrics[metric.__name__] += metric(y_valid_batch, valid_batch_output).detach().to_numpy().item()
                
            ##########################
            ### Store the results  ###
            ##########################
                  
            # Store the training and validation losses
            self.history["loss"].data = np.append(self.history["loss"].data, training_epoch_loss / n_training_steps)
            self.history["val_loss"].data = np.append(self.history["val_loss"].data, valid_epoch_loss / n_valid_steps)
            
            # Compute the average metrics
            for metric in metrics:
                # Compute the average of the metrics for the training and validation sets and store them
                self.history[metric.__name__].data = np.append(self.history[metric.__name__].data, train_metrics[metric.__name__] / n_training_steps)
                self.history[f"val_{metric.__name__}"].data = np.append(self.history[f"val_{metric.__name__}"].data, valid_metrics[metric.__name__] / n_valid_steps)
            
            #############################
            ### Display the progress  ###
            #############################
            
            # Display progress with metrics
            print(
                f"\rEpoch {self.epoch + 1}/{epochs} --> "
                f"loss: {self.history['loss'].data[-1]:.4f}"
                + "".join(
                    [f" - {metric.__name__.replace('_', ' ')}: {self.history[metric.__name__].data[-1]:.4f}" for metric in metrics]
                )
                + f" | Validation loss: {self.history['val_loss'].data[-1]:.4f}"
                + "".join(
                    [f" - Validation {metric.__name__.replace('_', ' ')}: {self.history[f'val_{metric.__name__}'].data[-1]:.4f}" for metric in metrics]
                ).ljust(50)
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
    
    def _forward(self, x: Tensor, batch_size: Optional[int] = None, verbose: bool = False) -> Tensor:
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
            batch_out = self.modules.forward(batch_out)
                
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
    
    
    def _lazy_init(self, x: Tensor) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, sequence length, embedding size)
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the number of modules is greater than 0
        assert len(self._modules.values()) > 0, "No modules in the neural network. Add modules to the model!"