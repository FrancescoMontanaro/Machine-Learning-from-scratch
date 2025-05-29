import numpy as np

from ..core import Tensor, Module


class Flatten(Module):
        
    ### Protected methods ###
    
    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass through the Flatten layer
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, ...)
        
        Returns:
        - Tensor: Flattened input data
        """
        
        # Extract the dimensions of the input data
        input_shape = x.shape()
        
        # Extract the batch size and number of features
        batch_size = input_shape[0]
        num_features = np.prod(input_shape[1:])
        
        # Flatten the input data
        return x.reshape((batch_size, num_features))