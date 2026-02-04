from typing import Literal

from ..core import Tensor, SingleOutputModule


class UpSampling2D(SingleOutputModule):
    
    ### Magic methods ###
    
    def __init__(self, size: tuple[int, int], interpolation: Literal["nearest"] = "nearest", *args, **kwargs) -> None:
        """
        Class constructor for UpSampling2D layer.
        
        Parameters:
        - size (tuple): scaling factors for the height and width dimensions
        - interpolation (str): interpolation method for upsampling. Default: "nearest"
        
        Raises:
        - AssertionError: if the size is not a tuple of 2 integers
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Check if the size is a tuple of 2 integers
        assert len(size) == 2, f"Size must be a tuple of 2 integers. Got: {size}"
        assert interpolation == "nearest", f"Interpolation method '{interpolation}' is not supported. Only 'nearest' is supported."
        
        # Initialize the parameters
        self.size = size
        self.interpolation = interpolation


    ### Protected methods ###
        
    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Function to compute the forward pass of the UpSampling2D layer.
        
        Parameters:
        - x (Tensor): input data. Shape: (Batch size, Height, Width, Channels)
        
        Returns:
        - Tensor: output data. Shape: (Batch size, Height x size[0], Width x size[1], Channels)
        
        Raises:
        - ValueError: if the interpolation method is not supported
        """
        
        # Extract the dimensions of the upsampling layer
        scale_height, scale_width = self.size
        
        # Upsampling by repeating the pixels in the height and width dimensions
        return x.repeat(scale_height, axis=1).repeat(scale_width, axis=2)
    
    
    def _lazy_init(self, x: Tensor, *args, **kwargs) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - x (Tensor): Features of the dataset
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape has a valid shape
        assert len(x.shape) == 4, f"Invalid input shape. Input must be a 4D array with shape (Batch size, Height, Width, Channels). Got shape: {x.shape}"