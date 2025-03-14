import numpy as np
from typing import Literal, Optional

from ..core import Tensor, Module
from ..activations import Activation


class Conv2D(Module):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        num_filters: int, 
        kernel_size: tuple[int, int], 
        padding: Literal["valid", "same"] = "valid", 
        stride: tuple[int, int] = (1, 1), 
        activation: Optional[Activation] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Class constructor for Conv2D layer.
        
        Parameters:
        - num_filters (int): number of filters in the convolutional layer
        - kernel_size (tuple[int, int]): size of the kernel
        - padding (Union[int, Literal["valid", "same"]]): padding to be applied to the input data. If "valid", no padding is applied. If "same", padding is applied to the input data such that the output size is the same as the input size.
        - stride (tuple[int, int]): stride of the kernel. First element is the stride along the height and second element is the stride along the width.
        - activation (Optional[Activation]): activation function to be applied to the output of the layer
        - name (str): name of the layer
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Check if the kernel size has a valid shape
        if len(kernel_size) != 2:
            raise ValueError("Kernel size must be a tuple of 2 integers.")
        
        # Initialize the parameters
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.activation = activation
        
        # Initializing the filters and bias
        self.filters: Tensor
        self.bias: Tensor
        
        # Initializing the input shape
        self.input_shape = None


    ### Public methods ###
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Function to compute the forward pass of the Conv2D layer.
        
        Parameters:
        - x (Tensor): input data. Shape: (Batch size, Height, Width, Channels)
        
        Returns:
        - Tensor: output data. Shape: (Batch size, Height, Width, Number of filters)
        
        Raises:
        - AssertionError: if the filters are not initialized
        """
        
        # Check if the input shape has a valid shape
        if len(x.shape()) != 4:
            # Raise an error if the input shape is not valid
            raise ValueError(f"Input must be a 4D array. The shape must be (Batch size, Height, Width, Channels). Got shape: {x.shape()}")
        
        # Extract the dimensions of the input data
        batch_size, input_height, input_width, num_channels = x.shape()
        
        # Save the input shape
        self.input_shape = (batch_size, input_height, input_width, num_channels)
        
        # Checking if the layer is initialized
        if not self.initialized:
            # Initialize the filters
            self.init_params(num_channels)
        
        # Assert that the filters are initialized
        assert isinstance(self.filters, Tensor), "Filters are not initialized. Please call the layer with some input data to initialize the filters."
        assert isinstance(self.bias, Tensor), "Bias is not initialized. Please call the layer with some input data to initialize the bias."
        
        # Extract the required dimensions for a better interpretation
        self.output_shape() # Compute the output shape of the Conv2D layer
        kernel_height, kernel_width = self.kernel_size # Shape of the kernel
        stride_height, stride_width = self.stride # Stride of the kernel
        
        # Apply padding to the input data
        if self.padding == "same":
            # Pad the input data
            x_padded = x.pad((
                (0, 0),
                (self.padding_top, self.padding_bottom),
                (self.padding_left, self.padding_right),
                (0, 0)
            ))
        else:
            # Set the padded input data as the input data
            x_padded = x
                
        # Extract the sliding windows from the input data
        patches = x_padded.sliding_window(
            window_shape = (kernel_height, kernel_width),
            axis = (1, 2)
        )
        
        # Apply the stride
        patches = patches[:, ::stride_height, ::stride_width, :, :, :]
        
        # Transpose the patches to have shape: (batch_size, output_height, output_width, kernel_height, kernel_width, num_channels)
        patches = patches.transpose((0, 1, 2, 4, 5, 3))

        # Compute the output of the Conv2D layer by applying the convolution operation
        out = patches.tensordot(self.filters, axes=([3, 4, 5], [1, 2, 3]))
        
        # Add the bias to the output
        out = out + self.bias
            
        # Apply the activation function
        if self.activation is not None:
            out = self.activation(out)
        
        # Return the output of the Conv2D layer
        return out

    
    def output_shape(self) -> tuple:
        """
        Function to compute the output shape of the Conv2D layer.
        
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
        batch_size, input_height, input_width, _ = self.input_shape
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        
        # Padding is applied to the input data
        if self.padding == 'same':
            # Compute the output shape of the Conv2D layer
            output_height = int(np.ceil(float(input_height) / float(stride_height)))
            output_width = int(np.ceil(float(input_width) / float(stride_width)))
            
            # Compute the padding values along the height and width
            pad_along_height = max((output_height - 1) * stride_height + kernel_height - input_height, 0)
            pad_along_width = max((output_width - 1) * stride_width + kernel_width - input_width, 0)
            
            # Compute the padding values
            self.padding_top = pad_along_height // 2
            self.padding_bottom = pad_along_height - self.padding_top
            self.padding_left = pad_along_width // 2
            self.padding_right = pad_along_width - self.padding_left
            
        # No padding is applied to the input data
        else:
            # Compute the output shape of the Conv2D layer
            output_height = int(np.floor((input_height - kernel_height) / stride_height) + 1)
            output_width = int(np.floor((input_width - kernel_width) / stride_width) + 1)
            
            # Set the padding values to 0
            self.padding_top = self.padding_bottom = self.padding_left = self.padding_right = 0
        
        # Compute the output shape of the Conv2D layer
        return (
            batch_size, # Batch size
            output_height, # Output height
            output_width,  # Output width
            self.num_filters # Number of filters
        )
    
    
    def init_params(self, num_channels: int) -> None:
        """
        Function to initialize the filters of the Conv2D layer.
        
        Parameters:
        - num_channels (int): number of channels in the input data
        """
        
        # Extract the dimensions of the kernel
        kernel_height, kernel_width = self.kernel_size
        
        # Initialize the filters with random values
        self.filters = Tensor(
            data = np.random.randn(self.num_filters, kernel_height, kernel_width, num_channels) / (kernel_height * kernel_width),
            requires_grad = True,
            is_parameter = True
        )
        
        # Initialize the bias with zeros
        self.bias = Tensor(
            data = np.zeros(self.num_filters),
            requires_grad = True,
            is_parameter = True
        )
        
        # Call the parent class method to set the layer as initialized
        super().init_params()