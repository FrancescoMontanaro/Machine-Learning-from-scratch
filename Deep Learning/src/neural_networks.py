import re
import numpy as np
from itertools import count
from typing import Callable

from .utils import *
from .layers import Layer
from .optimizers import Optimizer
from .loss_functions import LossFn


class FeedForward:
    
    ### Static attributes ###
    
    _ids = count(0) # Counter to keep track of the number of NeuralNetwork instances
    
    
    ### Magic methods ###
    
    def __init__(self, layers: list[Layer]) -> None:
        """
        Class constructor
        
        Parameters:
        - layers (list[_AbstractLayer]): List of Dense layers
        
        Raises:
        - ValueError: If the layer names are not unique
        """
        
        # List of layers
        self.layers = layers
        
        # Model settings
        self.training = False
        self.stop_training = False
        self.history = {}
        self.epoch = 0
        self.layers_outputs = {}
        self.layers_grads = {}
        
        # Get the count of how many NeuralNetwork instances have been created
        self.id = next(self._ids)
        
        # Set the name of the layers
        for i, layer in enumerate(self.layers):
            # Set the layer name if it is not set
            if not layer.name:
                # Get the class name of the layer
                layer.name = f"{re.sub(r'(?<!^)(?=[A-Z0-9])', '_', layer.__class__.__name__).lower()}_{i+1}"
                
        # Check if the layers have unique names
        if len(set([layer.name for layer in self.layers])) != len(self.layers):
            raise ValueError("Layer names must be unique!")
        
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Call method to initialize and execute the forward pass of the neural network
        
        Parameters:
        - x (np.ndarray): Input data. Shape: (Batch size, ...)
        
        Returns:
        - np.ndarray: Output of the neural network
        
        Raises:
        - ValueError: If the number of layers is 0
        """
        
        # Check if the number of layers is greater than 0
        if len(self.layers) == 0:
            raise ValueError("No layers in the neural network. Add layers to the model!")
        
        # Save the input dimension
        self.input_shape = x.shape
        
        # Set the model in evaluation mode
        self.eval()
        
        # Execute a first forward pass to initialize the parameters
        return self.forward(x)
        
        
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
        
        # Initialize the control variables
        self.history = {
            "loss": np.array([]),
            **{f"{metric.__name__}": np.array([]) for metric in metrics},
            "val_loss": np.array([]),
            **{f"val_{metric.__name__}": np.array([]) for metric in metrics}
        }
        self.stop_training = False
        self.epoch = 0
        n_steps = X_train.shape[0] // batch_size if batch_size < X_train.shape[0] else 1
        
        # Execute a first forward pass in evaluation mode to initialize the parameters and their shapes
        self(X_train[:1])
        
        # Add the optimizer to the layers
        for layer in self.layers:
            layer.optimizer = optimizer
        
        # Iterate over the epochs
        while self.epoch < epochs and not self.stop_training:
            
            ### Training phase ###
            
            # Set the model in training mode
            self.train()
            
            # Shuffle the dataset at the beginning of each epoch
            X_train_shuffled, Y_train_shuffled = shuffle_data(X_train, y_train)
            
            # Iterate over the batches
            epoch_loss = 0.0
            for step in range(n_steps):
                # Get the current batch of data
                X_batch = X_train_shuffled[step * batch_size:(step + 1) * batch_size]
                y_batch = Y_train_shuffled[step * batch_size:(step + 1) * batch_size]
                
                # Forward pass: Compute the output of the model
                batch_output = self.forward(X_batch)
                
                # Loss: Compute the error of the model
                loss = loss_fn(y_batch, batch_output)
                
                # Loss gradient: Compute the gradient of the loss with respect to the output of the model
                loss_grad = loss_fn.gradient(y_batch, batch_output)
                
                # Backward pass: Propagate the gradient through the model and update the parameters
                self.backward(loss_grad)
                
                # Update the epoch loss
                epoch_loss += loss
                
                # Display epoch progress
                print(f"\rEpoch {self.epoch + 1}/{epochs} ({round((((step + 1)/n_steps)*100), 2)}%) --> loss: {loss:.4f}", end="")
                
            ### Validation phase ###
                    
            # Set the model in evaluation mode
            self.eval()
            
            # Evaluate the model on the validation set
            validation_output = self.forward(X_valid)
            
            # Store the training and validation losses
            self.history["loss"] = np.append(self.history["loss"], epoch_loss / (X_train.shape[0] // batch_size))
            self.history["val_loss"] = np.append(self.history["val_loss"], loss_fn(y_valid, validation_output))
            
            # Compute the metrics
            for metric in metrics:
                # Compute the metric on the training set and validation set
                train_metric = metric(y_train, self.forward(X_train))
                valid_metric = metric(y_valid, validation_output)
                
                # Store the metrics
                self.history[metric.__name__] = np.append(self.history[metric.__name__], train_metric)
                self.history[f"val_{metric.__name__}"] = np.append(self.history[f"val_{metric.__name__}"], valid_metric)
            
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
            
            # Execute the callbacks
            for callback in callbacks:
                # Call the callback
                callback(self)
            
            # Increment the epoch counter
            self.epoch += 1
         
        # Return the history of the training   
        return self.history
        
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the neural network
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        
        Returns:
        - np.ndarray: Output of the neural network
        """
        
        # create a dictionary to store the output of each layer
        self.layers_outputs = {}
        
        # Copy the input
        out = np.copy(x)
        
        # Iterate over the layers
        for layer in self.layers:
            # Compute the output of the layer and pass it to the next one
            out = layer(out)
            
            # Store the output of the layer
            self.layers_outputs[layer.name] = out
            
        # Return the output of each layer of the neural network
        return out
    
    
    def backward(self, loss_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass of the neural network
        
        Parameters:
        - loss_grad (np.ndarray): Gradient of the loss with respect to the output of the neural network
        
        Returns:
        - np.ndarray: Gradient of the loss with respect to the input of the neural network
        """
        
        # Create a dictionary to store the gradient of each layer
        self.layers_grads = {}
        
        # Iterate over the layers in reverse order
        for layer in reversed(self.layers):
            # Store the gradient of the layer
            self.layers_grads[layer.name] = loss_grad
            
            # Compute the gradient of the loss with respect to the input of the layer
            loss_grad = layer.backward(loss_grad)
        
        # Return the gradient
        return loss_grad
    
    
    def train(self) -> None:
        """
        Method to set the model in training mode
        """
        
        # Set the training flag to True
        self.training = True
        
        # Iterate over the layers
        for layer in self.layers:
            # Set the layer in training mode
            layer.training = True

        
    def eval(self) -> None:
        """
        Method to set the model in evaluation mode
        """
        
        # Set the training flag to False
        self.training = False
        
        # Iterate over the layers
        for layer in self.layers:
            # Set the layer in evaluation mode
            layer.training = False
            
            
    def summary(self) -> None:
        """
        Method to display the summary of the neural network
        """
        
        def format_output(value: str, width: int) -> str:
            """
            Formats the output to fit within a specified width, splitting lines if necessary.
            
            Parameters:
            - value (str): The value to format
            - width (int): The width of the formatted output in characters
            
            Returns:
            - str: The formatted output
            """
            
            # Split the value by spaces to handle word wrapping
            words = value.split()
            formatted_lines = []
            current_line = ""

            # Iterate over the words
            for word in words:
                # Check if adding the word exceeds the width
                if len(current_line) + len(word) + 1 > width:  # +1 for space
                    # Add the current line to the list of lines
                    formatted_lines.append(current_line)
                    current_line = word
                else:
                    # Add the word to the current line
                    current_line += (" " + word) if current_line else word
            
            # Add the last line
            if current_line:
                formatted_lines.append(current_line)
                
            # Format each line to fit the specified width
            return "\n".join(line.ljust(width) for line in formatted_lines)
        
        # Display the header
        print(f"\nNeural Network (ID: {self.id})\n")
        header = f"{'Layer (type)':<40}{'Output Shape':<20}{'Trainable params #':<20}"
        print(f"{'-' * len(header)}")
        print(header)
        print(f"{'=' * len(header)}")

        # Iterate over the layers
        for idx, layer in enumerate(self.layers):
            # Composing the layer name
            layer_name = f"{layer.name} ({layer.__class__.__name__})"
            
            # Composing the output shape
            output_shape = "?"
            try:
                # Get the output shape of the layer
                output_shape = layer.output_shape() 
                
                # Format the output shape
                output_shape = f"({', '.join(str(dim) for dim in output_shape)})"
            except:
                pass
            
            # Composing the number of parameters
            num_params = "?"
            try:
                # Get the number of parameters of the layer
                num_params = layer.count_params()
            except:
                pass
            
            # format the output
            layer_name = format_output(layer_name, 40)
            output_shape = format_output(str(output_shape), 20)
            num_params = format_output(str(num_params), 20)
            
            # Display the layer information
            print(f"{layer_name:<40}{str(output_shape):<20}{str(num_params):<20}")
            if idx < len(self.layers) - 1 : print(f"{'-' * len(header)}")
            
        # Compute the total number of parameters
        total_params = "?"
        try:
            # Get the total number of parameters
            total_params = sum([layer.count_params() for layer in self.layers])
        except:
            pass
        
        # Display the footer 
        print(f"{'=' * len(header)}")
        print(f"Total trainable parameters: {total_params}")
        print(f"{'-' * len(header)}")