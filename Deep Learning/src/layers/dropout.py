import numpy as np

from ..core import Tensor, Module


class Dropout(Module):
    
    ### Magic methods ###
    
    def __init__(self, rate: float, *args, **kwargs) -> None:
        """
        Initialize the dropout layer.
        
        Parameters:
        - rate (float): The dropout rate.
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Initialize the dropout rate
        self.rate = rate
    
    
    ### Protected methods ###
    
    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass of the dropout layer.
        
        Parameters:
        - x (Tensor): The input tensor.
        
        Returns:
        - Tensor: The output tensor.
        """
        
        if self.training:
            # Generate a random mask
            mask = Tensor(
                data = np.random.rand(*x.shape) > self.rate, 
                requires_grad = False, 
                is_parameter = False
            )
            
            # Scale the output during training
            return x * mask / (1 - self.rate)
        else:
            # Return the output during inference
            return x