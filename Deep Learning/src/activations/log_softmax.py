import numpy as np

from ..core import Tensor
from .base import Activation
    

class LogSoftmax(Activation):
        
    ### Magic methods ###
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Compute the output of the log softmax activation function.
        
        Parameters:
        - x (Tensor): Input to the activation function
        
        Returns:
        - Tensor: Output of the activation function
        """
        
        return x.log_softmax()