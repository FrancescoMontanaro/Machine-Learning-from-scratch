from ..core import Tensor


class Activation:
    
    ### Magic methods ###

    def __call__(self, x: Tensor) -> Tensor:
        """
        Compute the output of the activation function.

        Parameters:
        - x (Tensor): Input to the activation function

        Returns:
        - Tensor: Output of the activation function
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method '__call__' is not implemented.")