from ..core import Tensor
from .base import Activation


class Tanh(Activation):
        
    ### Magic methods ###
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Compute the output of the hyperbolic tangent activation function.
        
        Parameters:
        - x (Tensor): Input to the activation function
        
        Returns:
        - Tensor: Output of the activation function
        """
        
        # Compute the hyperbolic tangent
        return x.tanh()