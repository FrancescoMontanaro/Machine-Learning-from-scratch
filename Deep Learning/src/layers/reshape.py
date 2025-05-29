from ..core import Tensor, Module


class Reshape(Module):
    
    ### Magic methods ###
    
    def __init__(self, shape: tuple, *args, **kwargs) -> None:
        """
        Class constructor for Reshape layer.
        
        Parameters:
        - shape (tuple): target shape of the input data
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Save the target shape
        self.target_shape = shape
    
    
    ### Protected methods ###
    
    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Function to compute the forward pass of the Reshape layer.
        
        Parameters:
        - x (Tensor): input data
        
        Returns:
        - Tensor: output data
        """
        
        # Extract the batch size
        batch = x.shape()[0]
        
        # Reshape the input data to the target shape
        # The batch size is kept the same
        return x.reshape((batch, *self.target_shape))