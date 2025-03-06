import re
import time
import math
import numpy as np
from itertools import count
from typing import Callable, Union, Optional

from .utils import *
from .layers import Layer
from .optimizers import Optimizer
from .loss_functions import LossFn


class Model:
    
    ### Static attributes ###
    
    _ids = count(0) # Counter to keep track of the number of NeuralNetwork instances
    
    
    ### Magic methods ###
    
    def __init__(self, modules: list[Union[Layer, 'Model']], name: Optional[str] = None) -> None:
        """
        Class constructor
        
        Parameters:
        - modules (list[Union[Layer, 'Model']]): List of modules (layers or models themselves) to be added to the model
        - name (Optional[str]): Name of the model
        
        Raises:
        - ValueError: If the modules names are not unique
        """
        
        # Get the count of how many NeuralNetwork instances have been created
        self.id = next(self._ids)
        
        # List of modules (layers or models) in the neural network
        self.modules = modules
        
        # Model settings
        self.training = False
        self.stop_training = False
        self.history = {}
        self.epoch = 0
        self.name = name if name else f"model_{self.id}"
        
        # Set the name of the layers
        for i, module in enumerate(self.modules):
            # Set the layer name if it is not set
            if not module.name and isinstance(module, Layer):
                # Get the class name of the layer
                module.name = f"{re.sub(r'(?<!^)(?=[A-Z0-9])', '_', module.__class__.__name__).lower()}_{i+1}"
                
        # Check if the modules have unique names
        if len(set([module.name for module in self.modules])) != len(self.modules):
            raise ValueError("Moduels names must be unique!")
        
        
    def __call__(self, x: np.ndarray, batch_size: Optional[int] = None, verbose: bool = False) -> np.ndarray:
        """
        Call method to initialize and execute the forward pass of the neural network
        
        Parameters:
        - x (np.ndarray): Input data.
        - batch_size (Optional[int]): Number of samples to use for each batch
        - verbose (bool): Flag to display the progress
        
        Returns:
        - np.ndarray: Output of the neural network
        
        Raises:
        - ValueError: If the number of modules is 0
        """
        
        # Check if the number of modules is greater than 0
        if len(self.modules) == 0:
            raise ValueError("No modules in the neural network. Add modules to the model!")
        
        # Save the input dimension
        self.input_shape = x.shape
        
        # Set the model in evaluation mode
        self.eval()
        
        # Execute a first forward pass to initialize the parameters
        return self.forward(
            x = x,
            batch_size = batch_size,
            verbose = verbose
        )
        
        
    ### Public methods ###      

    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        optimizer: Optimizer,
        loss_fn: LossFn,
        batch_size: int = 8, 
        epochs: int = 10,
        metrics: list[Callable] = [],
        callbacks: list[Callable] = []
    ) -> dict[str, np.ndarray]:
        """
        Method to train the neural network
        
        Parameters:
        - X_train (np.ndarray): Features of the training dataset. Shape: (samples, ...)
        - y_train (np.ndarray): Labels of the training dataset. Shape: (samples, ...)
        - X_valid (np.ndarray): Features of the validation dataset. Shape: (samples, ...)
        - y_valid (np.ndarray): Labels of the validation dataset. Shape: (samples, ...)
        - optimizer (Optimizer): Optimizer to update the parameters of the model
        - loss_fn (LossFn): Loss function to compute the error of the model
        - batch_size (int): Number of samples to use for each batch. Default is 32
        - epochs (int): Number of epochs to train the model. Default is 10
        - metrics (list[Callable]): List of metrics to evaluate the model. Default is an empty list
        - callbacks (list[Callback]): List of callbacks to execute
        
        Returns:
        - dict[str, np.ndarray]: Dictionary containing the training and validation losses
        """
        
        #######################
        ### Initializations ###
        #######################
        
        # Initialize the control variables
        self.history = {
            "loss": np.array([]),
            **{f"{metric.__name__}": np.array([]) for metric in metrics},
            "val_loss": np.array([]),
            **{f"val_{metric.__name__}": np.array([]) for metric in metrics}
        }
        self.epoch, self.stop_training = 0, False
        n_training_steps = math.ceil(X_train.shape[0] / batch_size) if batch_size < X_train.shape[0] else 1
        n_valid_steps = math.ceil(X_valid.shape[0] / batch_size) if batch_size < X_valid.shape[0] else 1
        
        # Execute a first forward pass in evaluation mode to initialize the parameters and their shapes
        self(X_train[:1])
        
        # Set the optimizer of the model
        self.set_optimizer(optimizer)
            
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
            X_train_shuffled, Y_train_shuffled = shuffle_data(X_train, y_train)
            
            # Iterate over the batches
            training_epoch_loss, elapsed_time = 0.0, 0.0
            train_metrics = {metric.__name__: 0.0 for metric in metrics}
            for training_step in range(n_training_steps):
                # Store the start time
                start_time = time.time()
                
                # Get the current batch of data
                X_training_batch = X_train_shuffled[training_step * batch_size:(training_step + 1) * batch_size]
                y_training_batch = Y_train_shuffled[training_step * batch_size:(training_step + 1) * batch_size]
                
                # Forward pass: Compute the output of the model
                training_batch_output = self.forward(X_training_batch)
                
                # Loss: Compute the error of the model
                training_loss = loss_fn(y_training_batch, training_batch_output)
                
                # Loss gradient: Compute the gradient of the loss with respect to the output of the model
                training_loss_grad = loss_fn.gradient(y_training_batch, training_batch_output)
                
                # Backward pass: Propagate the gradient through the model and update the parameters
                self.backward(training_loss_grad)
                
                # Update the epoch loss
                training_epoch_loss += training_loss
                
                # Compute the metrics
                for metric in metrics:
                    train_metrics[metric.__name__] += metric(y_training_batch, training_batch_output)
                    
                # Store the end time
                end_time = time.time()
                
                # Update the elapsed time
                elapsed_time += (end_time - start_time)
                
                # Compute the milliseconds per step
                ms_per_step = elapsed_time / (training_step + 1) * 1000
                
                # Display epoch progress
                print(f"\rEpoch {self.epoch + 1}/{epochs} ({round((((training_step + 1)/n_training_steps)*100), 2)}%) | {round(ms_per_step, 2)} ms/step --> loss: {training_loss:.4f}", end="")
            
            ##############################
            ### Start validation phase ###
            ##############################
                    
            # Set the model in evaluation mode
            self.eval()
            
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
                valid_epoch_loss += loss_fn(y_valid_batch, valid_batch_output)
                
                # Compute the metrics
                for metric in metrics:
                    valid_metrics[metric.__name__] += metric(y_valid_batch, valid_batch_output)
                
            ##########################
            ### Store the results  ###
            ##########################
                  
            # Store the training and validation losses
            self.history["loss"] = np.append(self.history["loss"], training_epoch_loss / n_training_steps)
            self.history["val_loss"] = np.append(self.history["val_loss"], valid_epoch_loss / n_valid_steps)
            
            # Compute the average metrics
            for metric in metrics:
                # Compute the average of the metrics for the training and validation sets and store them
                self.history[metric.__name__] = np.append(self.history[metric.__name__], train_metrics[metric.__name__] / n_training_steps)
                self.history[f"val_{metric.__name__}"] = np.append(self.history[f"val_{metric.__name__}"], valid_metrics[metric.__name__] / n_valid_steps) 
            
            #############################
            ### Display the progress  ###
            #############################
            
            # Display progress with metrics
            print(
                f"\rEpoch {self.epoch + 1}/{epochs} --> "
                f"loss: {self.history['loss'][-1]:.4f} "
                + " ".join(
                    [f"- {metric.__name__.replace('_', ' ')}: {self.history[metric.__name__][-1]:.4f}" for metric in metrics]
                )
                + f" | Validation loss: {self.history['val_loss'][-1]:.4f} "
                + " ".join(
                    [f"- Validation {metric.__name__.replace('_', ' ')}: {self.history[f'val_{metric.__name__}'][-1]:.4f}" for metric in metrics]
                )
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
         
        # Return the history of the training   
        return self.history
        
        
    def forward(self, x: np.ndarray, batch_size: Optional[int] = None, verbose: bool = False) -> np.ndarray:
        """
        Forward pass of the model
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        - batch_size (Optional[int]): Number of samples to use for each batch
        - verbose (bool): Flag to display the progress
        
        Returns:
        - np.ndarray: Output of the neural network
        """
        
        # Compute the number of steps to iterate over the batches
        # If the batch size is not provided, set it to 1
        num_steps = (math.ceil(x.shape[0] / batch_size) if batch_size < x.shape[0] else 1) if batch_size else 1
        
        # List to store the outputs of each module
        outputs = []
        
        # Variable to store the time per step
        elapsed_time = 0.0
        
        # Iterate over the batches
        for step in range(num_steps):
            # Store the start time
            start = time.time()
            
            # Get the current batch
            batch_out = x[step * batch_size:(step + 1) * batch_size] if batch_size else x
            
            # Iterate over the modules
            for module in self.modules:
                # Compute the output of the module and pass it to the next one
                batch_out = module(batch_out)
                
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
            
        # Concatenate the outputs of the batches to form an unique output and return it
        return np.concatenate(outputs, axis=0)
    
    
    def backward(self, loss_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass of the model
        
        Parameters:
        - loss_grad (np.ndarray): Gradient of the loss with respect to the output
        
        Returns:
        - np.ndarray: Gradient of the loss with respect to the input
        """
        
        # Iterate over the modules in reverse order
        for module in reversed(self.modules):
            # Compute the gradient of the loss with respect to the input of the module
            loss_grad = module.backward(loss_grad)
        
        # Return the gradient
        return loss_grad
    
    
    def train(self) -> None:
        """
        Method to set the model in training mode
        """
        
        # Set the training flag to True
        self.training = True
        
        # Iterate over the modules
        for module in self.modules:
            # Set the module in training mode
            module.training = True

        
    def eval(self) -> None:
        """
        Method to set the model in evaluation mode
        """
        
        # Set the training flag to False
        self.training = False
        
        # Iterate over the modules
        for module in self.modules:
            # Set the module in evaluation mode
            module.training = False
            
            
    def summary(self) -> None:
        """
        Method to display the summary of the model
        """
        
        # Regular expression pattern to check if the model name is the default one
        model_name_pattern = re.compile(r'^model_\d+$')
        
        # Display the header
        print(f"\n{'Model' if re.match(model_name_pattern, self.name) else self.name} (ID: {self.id})\n")
        header = f"{'Module (type)':<40}{'Output Shape':<20}{'Trainable params #':<20}"
        print(f"{'-' * len(header)}")
        print(header)
        print(f"{'=' * len(header)}")

        # Iterate over the modules
        for idx, module in enumerate(self.modules):
            # Composing the module name
            module_name = f"{module.name} ({module.__class__.__name__})"
            
            # Composing the output shape
            output_shape = "?"
            try:
                # Get the output shape of the module
                output_shape = module.output_shape() 
                
                # Format the output shape
                output_shape = f"({', '.join(str(dim) for dim in output_shape)})"
            except:
                pass
            
            # Composing the number of parameters
            num_params = "?"
            try:
                # Get the number of parameters of the module
                num_params = module.count_params()
            except:
                pass
            
            # format the output
            module_name = format_summary_output(module_name, 40)
            output_shape = format_summary_output(str(output_shape), 20)
            num_params = format_summary_output(str(num_params), 20)
            
            # Display the module information
            print(f"{module_name:<40}{str(output_shape):<20}{str(num_params):<20}")
            if idx < len(self.modules) - 1 : print(f"{'-' * len(header)}")
            
        # Compute the total number of parameters
        total_params = "?"
        try:
            # Get the total number of parameters
            total_params = self.count_params()
        except:
            pass
        
        # Display the footer 
        print(f"{'=' * len(header)}")
        print(f"Total trainable parameters: {total_params}")
        print(f"{'-' * len(header)}")
        
        
    def set_optimizer(self, optimizer: Optimizer) -> None:
        """
        Method to set the optimizer of the model
        
        Parameters:
        - optimizer (Optimizer): Optimizer to update the parameters of the model
        """
        
        # Iterate over the modules of the model
        for module in self.modules:
            # Recursively set the optimizer of the module
            module.set_optimizer(optimizer)
                
                
    def count_params(self) -> int:
        """
        Method to count the number of parameters of the model
        
        Returns:
        - int: Number of parameters of the model
        """
        
        # Get the number of parameters of each module
        return sum([module.count_params() for module in self.modules])
    
    
    def output_shape(self) -> tuple:
        """
        Method to get the output shape of the model
        
        Returns:
        - tuple[int]: Output shape of the model
        """
        
        # Get the output shape of the last module by calling the output_shape method
        # This will evetually be called recursively until the last the module is a layer
        # which returns the output shape
        return self.modules[-1].output_shape()