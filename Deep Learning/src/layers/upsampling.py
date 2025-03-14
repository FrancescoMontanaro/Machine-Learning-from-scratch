import numpy as np
from typing import Literal, Optional

from .base import Layer


class UpSampling2D(Layer):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        size: tuple[int, int],
        interpolation: Literal["nearest"] = "nearest",
        name: Optional[str] = None
    ) -> None:
        """
        Class constructor for UpSampling2D layer.
        
        Parameters:
        - size (tuple): size of the upsampling operation
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
        
        # Initializing the input shape
        self.input_shape = None
    
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Function to compute the forward pass of the UpSampling2D layer.
        It initializes the filters if not initialized and computes the forward pass.
        
        Parameters:
        - x (np.ndarray): input data. Shape: (Batch size, Height, Width, Channels)
        
        Returns:
        - np.ndarray: output data. Shape: (Batch size, Height x size[0], Width x size[1], Channels)
        
        Raises:
        - ValueError: if the input shape is not a tuple of 4 integers
        """
        
        # Check if the input shape has a valid shape
        if len(x.shape) != 4:
            raise ValueError(f"Input must be a 4D array. The shape must be (Batch size, Height, Width, Channels). Got shape: {x.shape}")
        
        # Extract the dimensions of the input data
        batch_size, input_height, input_width, num_channels = x.shape
        
        # Save the input shape
        self.input_shape = (batch_size, input_height, input_width, num_channels)
        
        # Checking if the layer is initialized
        if not self.initialized:
            # Initialize the filters
            self.init_params(num_channels)
            
        # Compute the forward pass
        return self.forward(x)


    ### Public methods ###
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Function to compute the forward pass of the UpSampling2D layer.
        
        Parameters:
        - x (np.ndarray): input data. Shape: (Batch size, Height, Width, Channels)
        
        Returns:
        - np.ndarray: output data. Shape: (Batch size, Height x size[0], Width x size[1], Channels)
        
        Raises:
        - ValueError: if the interpolation method is not supported
        """
        
        # Save the input data
        self.x = x
        
        # Extract the dimensions of upsampling operation
        scale_height, scale_width = self.size
        
        if self.interpolation == "nearest":
            # Upsampling by repeating the pixels in the height and width dimensions
            output = np.repeat(np.repeat(x, scale_height, axis=1), scale_width, axis=2)
        else:
            # Raise an error if the interpolation method is not supported
            raise ValueError(f"Interpolation method '{self.interpolation}' is not supported.") 
            
        return output
    
    
    def backward(self, loss_gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass of the layer (layer i)
        
        Parameters:
        - loss_gradient (np.ndarray): Gradient of the loss with respect to the output of the layer: dL/dO_i
        
        Returns:
        - np.ndarray: Gradient of the loss with respect to the input of the layer: dL/dX_i ≡ dL/dO_{i-1}
        
        Raises:
        - AssertionError: if the input shape is not set
        - ValueError: if the interpolation method is not supported
        """
        
        # Assert that the input shape is set
        assert self.input_shape is not None, "Input shape is not set. Please call the layer with some input data to set the input shape."
        
        # Extract the dimensions of the input data and the size of the upsampling operation
        batch_size, input_height, input_width, n_channels = self.input_shape
        scale_height, scale_width = self.size
        
        # Bilinear interpolation
        if self.interpolation == "nearest":
            # Reshape the loss gradient to the shape of the input data and sum over the repeated dimensions
            d_input = loss_gradient.reshape(batch_size, input_height, scale_height, input_width, scale_width, n_channels)
            d_input = d_input.sum(axis=(2, 4))
        else:
            # Raise an error if the interpolation method is not supported
            raise ValueError(f"Interpolation method '{self.interpolation}' is not supported.")

        # Return the gradient
        return d_input
    
    
    def output_shape(self) -> tuple:
        """
        Function to compute the output shape of the UpSampling2D layer.
        
        Returns:
        - tuple: shape of the output data
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
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