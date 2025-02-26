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
            # Calcola padding per l'altezza
            top_padding = int(np.floor(self.padding_height / 2))
            bottom_padding = int(np.ceil(self.padding_height / 2))
            
            # Calcola padding per la larghezza
            left_padding = int(np.floor(self.padding_width / 2))
            right_padding = int(np.ceil(self.padding_width / 2))
            
            # Pad the input data
            x = np.pad(x, ((0, 0), (top_padding, bottom_padding), (left_padding, right_padding), (0, 0)), mode="constant")
            
        # Cache for max positions
        self.cache = np.zeros_like(x, dtype=bool) 
        
        # Initialize the output
        output = np.zeros((batch_size, output_height, output_width, n_channels))
        
        # Iterate over the input data
        for i in range(output_height):
            # Iterate over the width
            for j in range(output_width):
                # Extract the current pooling region
                x_slice = x[:, i * stride_height : i * stride_height + pool_height, j * stride_width : j * stride_width + pool_width, :]
                
                # Compute the max value and the mask
                max_mask = (x_slice == np.max(x_slice, axis=(1, 2), keepdims=True))
                output[:, i, j, :] = np.max(x_slice, axis=(1, 2))
                
                # Save the mask in the cache
                self.cache[:, i * stride_height : i * stride_height + pool_height, j * stride_width : j * stride_width + pool_width, :] |= max_mask
                
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
        pool_height, pool_width = self.size
        stride_height, stride_width = self.stride
        
        # Initialize the gradient of the input with zeros
        d_input = np.zeros(self.input_shape)
        
        # Iterate over the height dimension
        for i in range(output_height):
            # Iterate over the width dimension
            for j in range(output_width):
                # Create a slice of the cached mask for the current pooling region
                mask_slice = self.cache[:, i * stride_height : i * stride_height + pool_height, j * stride_width : j * stride_width + pool_width, :]
                
                # Distribute the gradient only to the positions that were the max in the forward pass
                d_input[:, i * stride_height : i * stride_height + pool_height, j * stride_width : j * stride_width + pool_width, :] += mask_slice * loss_gradient[:, i, j, :][:, np.newaxis, np.newaxis, :]
           
        return d_input # dL/dX_i ≡ dL/dO_{i-1}
        
        
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
            self.padding_height = (np.ceil((input_height - pool_height) / stride_height) * stride_height + pool_height - input_height) / 2
            self.padding_width = (np.ceil((input_width - pool_width) / stride_width) * stride_width + pool_width - input_width) / 2
            
            # Compute the output shape with padding
            output_height = int(((input_width - pool_height + 2 * self.padding_height) / stride_height) + 1)
            output_width = int(((input_width - pool_width + 2 * self.padding_width) / stride_width) + 1)
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