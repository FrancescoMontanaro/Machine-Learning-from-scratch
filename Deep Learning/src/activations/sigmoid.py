from ..core import Tensor
from .base import Activation
    
    
class Sigmoid(Activation):
    
    ### Magic methods ###
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Compute the output of the sigmoid activation function.
        
        Parameters:
        - x (Tensor): Input to the activation function
        
        Returns:
        - Tensor: Output of the activation function
        """
        
        # Compute the sigmoid
        return x.sigmoid()