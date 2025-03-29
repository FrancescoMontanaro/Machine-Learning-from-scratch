from ..core import Tensor, Module


class Input(Module):
    
    ### Public methods ###
    
    def _forward(self, x: Tensor) -> Tensor:
        """
        Function to compute the forward pass of the Input layer.
        
        Parameters:
        - x (Tensor): input data
        
        Returns:
        - Tensor: output data
        """
        
        # Return the input data as it is
        return x