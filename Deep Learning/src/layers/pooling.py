import numpy as np
from typing import Optional, Literal

from ..core import Tensor, Module


class MaxPool2D(Module):
    
    ### Magic methods ###
    
    def __init__(self, size: tuple[int, int] = (2, 2), stride: Optional[tuple[int, int]] = None, padding: Literal["valid", "same"] = "valid", name: Optional[str] = None) -> None:
        """
        Class constructor for the MaxPool2D layer
        
        Parameters:
        - size (tuple): Size of the pooling window
        - stride (tuple): Stride of the pooling window
        - name (str): Name of the layer
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Save the pooling window size, stride and padding
        self.size = size
        self.stride = stride if stride is not None else size
        self.padding = padding
        
        
        
    ### Public methods ###
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Method to perform the forward pass through the MaxPool2D layer
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, height, width, channels)
        
        Returns:
        - Tensor: Output of the layer after the forward pass
        """
        
        # Check if the input shape has a valid shape
        if len(x.shape()) != 4:
            raise ValueError(f"Input must be a 4D array. The shape must be (Batch size, Height, Width, Channels). Got shape: {x.shape()}")
        
        # Extract the dimensions of the input data
        batch_size, input_height, input_width, num_channels = x.shape()
        
        # Save the input shape
        self.input_shape = (batch_size, input_height, input_width, num_channels)
        
        # Initialize the layer params
        if not self.initialized:
            self.init_params()
            
        # Apply padding to the input data
        if self.padding == "same":
            # Pad the input data
            x = x.pad((
                (0, 0),
                (self.top_padding, self.bottom_padding),
                (self.left_padding, self.right_padding),
                (0, 0)
            ))
        
        # Perform the max pooling 2D operation
        out = x.max_pool_2d(self.size, self.stride)
                
        # Return the output
        return out
        
        
    def output_shape(self) -> tuple:
        """
        Function to compute the output shape of the layer.
        
        Returns:
        - tuple: shape of the output data
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
        # Call the parent class method to check if the layer is initialized
        super().output_shape()
        
        # Assert that the input shape is set
        assert self.input_shape is not None, "Input shape is not set. Please call the layer with some input data to set the input shape."
        
        # Extract the dimensions
        batch_size, input_height, input_width, n_channels = self.input_shape
        pool_width, pool_height = self.size
        stride_height, stride_width = self.stride
        
        # Compute the output shape
        
        if self.padding == "same":
            # Compute the padding
            padding_height = (np.ceil((input_height - pool_height) / stride_height) * stride_height + pool_height - input_height) / 2
            padding_width = (np.ceil((input_width - pool_width) / stride_width) * stride_width + pool_width - input_width) / 2
            
            # Compute the output shape with padding
            output_height = int(((input_width - pool_height + 2 * padding_height) / stride_height) + 1)
            output_width = int(((input_width - pool_width + 2 * padding_width) / stride_width) + 1)
            
            # Compute the padding values
            self.top_padding = int(np.floor(padding_height / 2))
            self.bottom_padding = int(np.ceil(padding_height / 2))
            self.left_padding = int(np.floor(padding_width / 2))
            self.right_padding = int(np.ceil(padding_width / 2))
            
        else:
            # Compute the output shape without padding
            output_height = (input_height - pool_height) // stride_height + 1
            output_width = (input_width - pool_width) // stride_width + 1
    
        return (
            batch_size, # Batch size
            output_height, # Height
            output_width, # Width
            n_channels # Number of channels
        )