import numpy as np
from typing import Optional, Literal

from ..core import Tensor, SingleOutputModule


class MaxPool2D(SingleOutputModule):
    
    ### Magic methods ###
    
    def __init__(self, size: tuple[int, int] = (2, 2), stride: Optional[tuple[int, int]] = None, padding: Literal["valid", "same"] = "valid", *args, **kwargs) -> None:
        """
        Class constructor for the MaxPool2D layer
        
        Parameters:
        - size (tuple): Size of the pooling window
        - stride (tuple): Stride of the pooling window
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Save the pooling window size, stride and padding
        self.size = size
        self.stride = stride if stride is not None else size
        self.padding = padding
        

    ### Protected methods ###
    
    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Method to perform the forward pass through the MaxPool2D layer
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, height, width, channels)
        
        Returns:
        - Tensor: Output of the layer after the forward pass
        """
            
        # Apply padding to the input data
        if self.padding == "same":
            # Extract the input shape, pooling window size and stride
            _, input_height, input_width, _ = x.shape
            pool_width, pool_height = self.size
            stride_height, stride_width = self.stride
            
            # Compute the padding along the height and width
            padding_height = (np.ceil((input_height - pool_height) / stride_height) * stride_height + pool_height - input_height) / 2
            padding_width = (np.ceil((input_width - pool_width) / stride_width) * stride_width + pool_width - input_width) / 2
            
            # Compute the padding values along all sides
            top_padding = int(np.floor(padding_height / 2))
            bottom_padding = int(np.ceil(padding_height / 2))
            left_padding = int(np.floor(padding_width / 2))
            right_padding = int(np.ceil(padding_width / 2))
            
            # Pad the input data
            x = x.pad((
                (0, 0),
                (top_padding, bottom_padding),
                (left_padding, right_padding),
                (0, 0)
            ))
        
        # Perform the max pooling 2D operation
        out = x.max_pool_2d(self.size, self.stride)
                
        # Return the output
        return out
        
    
    def _lazy_init(self, x: Tensor, *args, **kwargs) -> None:
        """
        Method to initialize the parameters of the layer
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, height, width, channels)
        
        Raises:
        - AssertionError: if the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape) == 4, f"Input must be a 4D array. The shape must be (Batch size, Height, Width, Channels). Got shape: {x.shape}"