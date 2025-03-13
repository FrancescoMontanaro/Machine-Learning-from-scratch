import numpy as np
from typing import Optional

from ..core import Tensor, Module


class Reshape(Module):
    
    ### Magic methods ###
    
    def __init__(self, shape: tuple, name: Optional[str] = None) -> None:
        """
        Class constructor for Reshape layer.
        
        Parameters:
        - shape (tuple): target shape of the input data
        - name (str): name of the layer
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Save the target shape
        self.target_shape = shape
    
    
    ### Public methods ###
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Function to compute the forward pass of the Reshape layer.
        
        Parameters:
        - x (Tensor): input data
        
        Returns:
        - Tensor: output data
        """
        
        # Save the input shape
        self.input_shape = x.shape()
        
        # Checking if the layer is initialized
        if not self.initialized:
            # Initialize the filters
            self.init_params()
        
        # Extract the batch size
        batch = self.input_shape[0]
        
        # Reshape the input data to the target shape
        # The batch size is kept the same
        return x.reshape((batch, *self.target_shape))
    
    
    def output_shape(self) -> tuple:
        """
        Method to get the output shape of the layer.
        
        Returns:
        - tuple: output shape of the layer
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
        # Call the parent class method to check if the layer is initialized
        super().output_shape()
        
        # Assert that the input shape is set
        assert self.input_shape is not None, "Input shape is not set. Please call the layer with some input data to set the input shape."
        
        # Extract the batch size
        batch = self.input_shape[0]
        
        # Return the output shape
        # The batch size is kept the same
        return (batch, *self.target_shape)