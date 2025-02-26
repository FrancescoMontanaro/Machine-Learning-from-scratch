import numpy as np
from typing import Optional

from .base import Layer


class Flatten(Layer):
    
    ### Magic methods ###
    
    def __init__(self, name: Optional[str] = None) -> None:
        """
        Class constructor 
        
        Parameters:
        - name (str): Name of the layer
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Initialize the input shape
        self.input_shape = None
        
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the Flatten layer
        
        Parameters:
        - x (np.ndarray): Input data. Shape: (Batch size, ...)
        
        Returns:
        - np.ndarray: Flattened input data
        """
        
        # Save the input shape
        self.input_shape = x.shape
        
        # Initialize the layer params
        if not self.initialized:
            self.init_params()
        
        # Perform the forward pass
        return self.forward(x)
        
        
    ### Public methods ###
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the Flatten layer
        
        Parameters:
        - x (np.ndarray): Input data. Shape: (Batch size, ...)
        
        Returns:
        - np.ndarray: Flattened input data
        """
    
        # Extract the output dimensions
        batch_size, num_features = self.output_shape()
        
        # Flatten the input data
        return x.reshape(batch_size, num_features)
    
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass through the Flatten layer
        
        Parameters:
        - grad (np.ndarray): Gradient of the loss with respect to the output of the Flatten layer
        
        Returns:
        - np.ndarray: Gradient of the loss with respect to the input of the Flatten layer
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
        # Assert that the input shape is set
        assert self.input_shape is not None, "Input shape is not set. Please call the layer with some input data to set the input shape."
        
        # Reshape the gradient
        return grad.reshape(self.input_shape)
    
    
    def output_shape(self) -> tuple:
        """
        Function to compute the output shape of the layer.
        
        Returns:
        - tuple: shape of the output data
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
        # Assert that the input shape is set
        assert self.input_shape is not None, "Input shape is not set. Please call the layer with some input data to set the input shape."
        
        # Extract the dimensions
        batch_size = self.input_shape[0]
        num_features = np.prod(self.input_shape[1:])
        
        # Compute the output shape
        return (batch_size, num_features)