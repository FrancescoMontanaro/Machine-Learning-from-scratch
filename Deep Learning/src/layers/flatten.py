import numpy as np
from typing import Optional

from ..core import Tensor, Module


class Flatten(Module):
    
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
        
        
    ### Public methods ###
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Flatten layer
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, ...)
        
        Returns:
        - Tensor: Flattened input data
        """
        
        # Save the input shape
        self.input_shape = x.shape()
        
        # Initialize the layer params
        if not self.initialized:
            self.init_params()
    
        # Extract the output dimensions
        batch_size, num_features = self.output_shape()
        
        # Flatten the input data
        return x.reshape((batch_size, num_features))
    
    
    def output_shape(self) -> tuple:
        """
        Function to compute the output shape of the layer.
        
        Returns:
        - tuple: shape of the output data
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
        # Call the parent class method to check if the layer is initialized
        super().output_shape()
        
        # Assert that the input shape is set
        assert self.input_shape is not None, "Input shape is not set. Please call the layer with some input data to set the input shape."
        
        # Extract the dimensions
        batch_size = self.input_shape[0]
        num_features = np.prod(self.input_shape[1:])
        
        # Compute the output shape
        return (batch_size, num_features)