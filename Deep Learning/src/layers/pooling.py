import numpy as np
from typing import Optional, Literal

from .base import Layer


class MaxPool2D(Layer):
    
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
        
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Method to perform the forward pass through the MaxPool2D layer
        
        Parameters:
        - x (np.ndarray): Input data. Shape: (Batch size, height, width, channels)
        
        Returns:
        - np.ndarray: Output of the layer after the forward pass
        """
        
        # Check if the input shape has a valid shape
        if len(x.shape) != 4:
            raise ValueError(f"Input must be a 4D array. The shape must be (Batch size, Height, Width, Channels). Got shape: {x.shape}")
        
        # Extract the dimensions of the input data
        batch_size, input_height, input_width, num_channels = x.shape
        
        # Save the input shape
        self.input_shape = (batch_size, input_height, input_width, num_channels)
        
        # Initialize the layer params
        if not self.initialized:
            self.init_params()
        
        # Perform the forward pass
        return self.forward(x)
        
        
    ### Public methods ###
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Method to perform the forward pass through the MaxPool2D layer
        
        Parameters:
        - x (np.ndarray): Input data. Shape: (Batch size, height, width, channels)
        
        Returns:
        - np.ndarray: Output of the layer after the forward pass
        """
        
        # Extract the required dimensions for better readability
        batch_size, output_height, output_width, n_channels = self.output_shape()
        pool_height, pool_width = self.size
        stride_height, stride_width = self.stride
        
        # Apply the padding
        if self.padding == "same":
            # Pad the input data
            x = np.pad(
                x, 
                (
                    (0, 0), 
                    (self.top_padding, self.bottom_padding), 
                    (self.left_padding, self.right_padding), 
                    (0, 0)
                ), 
                mode="constant"
            )
            
        # Save the input data for the backward pass
        self.x = x
        
        # Extract the patches from the input data
        patches = np.lib.stride_tricks.sliding_window_view(
            x, 
            window_shape = (pool_height, pool_width), 
            axis = (1, 2)  # type: ignore
        )
        
        # Apply the stride to the patches
        patches = patches[:, ::stride_height, ::stride_width, :, :, :]
        
        # Reshape the patches to have dimensions: (batch_size, output_height, output_width, pool_height, pool_width, n_channels)
        patches = patches.transpose(0, 1, 2, 4, 5, 3)

        # For each patch, compute the max value
        output = np.max(patches, axis=(3, 4)) # shape: (batch_size, output_height, output_width, n_channels)

        # Reshape the patches to have dimensions: (batch_size, output_height, output_width, pool_height * pool_width, n_channels)
        flat_patches = patches.reshape(batch_size, output_height, output_width, pool_height * pool_width, n_channels)
        
        # Save the indices of the max values for the backward pass
        self.cache = np.argmax(flat_patches, axis=3) 
                
        return output
    
    
    def backward(self, loss_gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass for the MaxPool2D layer (Layer i)
        
        Parameters:
        - loss_gradient (np.ndarray): Gradient of the loss with respect to the output of this layer: dL/dO_i
        
        Returns:
        - np.ndarray: Gradient of the loss with respect to the input of this layer: dL/dX_i ≡ dL/dO_{i-1}
        """
        
        # Extract the required dimensions for better readability
        batch_size, output_height, output_width, n_channels = loss_gradient.shape
        _, pool_width = self.size
        stride_height, stride_width = self.stride
        
        # Initialize the gradient of the loss with respect to the input
        d_input = np.zeros_like(self.x)
        
        # Flatten cache to have shape (batch_size*out_height*out_width, n_channels)
        argmax_flat = self.cache.reshape(-1, n_channels)
        
        # Decompose that offset into (row_offset, col_offset) and flatten them
        row_offset = (argmax_flat // pool_width).ravel()
        col_offset = (argmax_flat %  pool_width).ravel()

        # Create coordinate grids for the batch, output height, and output width and flatten them
        b_idx, oh_idx, ow_idx = np.indices((batch_size, output_height, output_width), sparse=False)
        b_idx, oh_idx, ow_idx = b_idx.ravel(), oh_idx.ravel(), ow_idx.ravel()

        # Replicate the batch index for each channel
        b_expanded  = np.repeat(b_idx,  n_channels)
        oh_expanded = np.repeat(oh_idx, n_channels)
        ow_expanded = np.repeat(ow_idx, n_channels)

        # The final row, col in the padded d_input
        row_idx = oh_expanded * stride_height + row_offset
        col_idx = ow_expanded * stride_width  + col_offset

        # Replicate the channel index for each batch, output height, and output width
        c_expanded = np.tile(np.arange(n_channels), b_idx.shape[0])

        # Flatten the gradient and add the values to the correct indices
        grad_flat = loss_gradient.reshape(-1, n_channels).ravel()
        np.add.at(d_input, (b_expanded, row_idx, col_idx, c_expanded), grad_flat)

        # Remove padding if applied during the forward pass
        if self.padding == "same":
            d_input = d_input[
                :, 
                self.top_padding : d_input.shape[1] - self.bottom_padding,
                self.left_padding: d_input.shape[2] - self.right_padding,
                :
            ]
        
        return d_input
        
        
    def output_shape(self) -> tuple:
        """
        Function to compute the output shape of the layer.
        
        Returns:
        - tuple: shape of the output data
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
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