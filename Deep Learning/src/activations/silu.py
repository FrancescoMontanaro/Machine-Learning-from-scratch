from ..core import Tensor
from .base import Activation


class SiLU(Activation):

    ### Magic methods ###
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Compute the output of the SiLU activation function.
        
        Parameters:
        - x (Tensor): Input to the activation function
        
        Returns:
        - Tensor: Output of the activation function
        """

        # Compute the SiLU activation
        return x * x.sigmoid()