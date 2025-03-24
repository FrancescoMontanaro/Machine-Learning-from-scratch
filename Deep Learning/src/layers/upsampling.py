from typing import Literal, Optional

from ..core import Tensor, Module


class UpSampling2D(Module):
    
    ### Magic methods ###
    
    def __init__(self, size: tuple[int, int], interpolation: Literal["nearest"] = "nearest", name: Optional[str] = None) -> None:
        """
        Class constructor for UpSampling2D layer.
        
        Parameters:
        - size (tuple): scaling factors for the height and width dimensions
        - interpolation (str): interpolation method for upsampling. Default: "nearest"
        - name (str): name of the layer
        
        Raises:
        - ValueError: if the size is not a tuple of 2 integers
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Check if the size is a tuple of 2 integers
        if len(size) != 2:
            raise ValueError(f"Size must be a tuple of 2 integers. Got: {size}")
        
        # Initialize the parameters
        self.size = size
        self.interpolation = interpolation


    ### Public methods ###
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Function to compute the forward pass of the UpSampling2D layer.
        
        Parameters:
        - x (Tensor): input data. Shape: (Batch size, Height, Width, Channels)
        
        Returns:
        - Tensor: output data. Shape: (Batch size, Height x size[0], Width x size[1], Channels)
        
        Raises:
        - ValueError: if the input shape is not a tuple of 4 integers
        - ValueError: if the interpolation method is not supported
        """
        
        # Check if the input shape has a valid shape
        if len(x.shape()) != 4:
            raise ValueError(f"Input must be a 4D array. The shape must be (Batch size, Height, Width, Channels). Got shape: {x.shape()}")
        
        # Extract the dimensions of the input data and the dimensions of the upsampling operation
        self.input_shape = x.shape()
        scale_height, scale_width = self.size
        num_channels = self.input_shape[-1]
        
        # Checking if the layer is initialized
        if not self.initialized:
            # Initialize the filters
            self.init_params(num_channels)
        
        # Nearest neighbor interpolation
        if self.interpolation == "nearest":
            # Upsampling by repeating the pixels in the height and width dimensions
            output = x.repeat(scale_height, axis=1).repeat(scale_width, axis=2)
        else:
            # Raise an error if the interpolation method is not supported
            raise ValueError(f"Interpolation method '{self.interpolation}' is not supported.") 
            
        return output
    
    
    def output_shape(self) -> tuple:
        """
        Function to compute the output shape of the UpSampling2D layer.
        
        Returns:
        - tuple: shape of the output data
        
        Raises:
        - AssertionError: if the input shape is not set
        """
                
        # Call the parent class method to check if the layer is initialized
        super().output_shape()
        
        # Assert that the input shape is set
        assert self.input_shape is not None, "Input shape is not set. Please call the layer with some input data to set the input shape."
        
        # Unpack the input shape
        batch_size, input_height, input_width, num_channels = self.input_shape
        
        # Compute the output shape of the UpSampling2D layer
        return (
            batch_size,
            input_height * self.size[0],
            input_width * self.size[1],
            num_channels
        )