from ..core import Tensor, SingleOutputModule


class Input(SingleOutputModule):
    
    ### Protected methods ###
    
    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Function to compute the forward pass of the Input layer.
        
        Parameters:
        - x (Tensor): input data
        
        Returns:
        - Tensor: output data
        """
        
        # Return the input data as it is
        return x