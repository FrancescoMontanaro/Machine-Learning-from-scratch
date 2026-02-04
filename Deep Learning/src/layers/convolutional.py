import numpy as np
from typing import Literal, Optional, Tuple

from ..activations import Activation
from ..core import Tensor, SingleOutputModule


class Conv2D(SingleOutputModule):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        num_filters: int, 
        kernel_size: tuple[int, int], 
        padding: Literal["valid", "same"] = "valid", 
        stride: tuple[int, int] = (1, 1), 
        activation: Optional[Activation] = None,
        *args, 
        **kwargs
    ) -> None:
        """
        Class constructor for Conv2D layer.
        
        Parameters:
        - num_filters (int): number of filters in the convolutional layer
        - kernel_size (tuple[int, int]): size of the kernel
        - padding (Union[int, Literal["valid", "same"]]): padding to be applied to the input data. If "valid", no padding is applied. If "same", padding is applied to the input data such that the output size is the same as the input size.
        - stride (tuple[int, int]): stride of the kernel. First element is the stride along the height and second element is the stride along the width.
        - activation (Optional[Activation]): activation function to be applied to the output of the layer
        
        Raises:
        - AssertionError: if the kernel size is not a tuple of 2 integers
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Check if the kernel size has a valid shape
        assert len(kernel_size) == 2, f"Kernel size must be a tuple of 2 integers. Got: {kernel_size}"
        
        # Initialize the parameters
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.activation = activation
        
        # Initializing the filters and bias
        self.filters: Tensor
        self.bias: Tensor


    ### Protected methods ###
        
    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Function to compute the forward pass of the Conv2D layer.
        
        Parameters:
        - x (Tensor): input data. Shape: (Batch size, Height, Width, Channels)
        
        Returns:
        - Tensor: output data. Shape: (Batch size, Height, Width, Number of filters)
        """
        
        # Apply padding to the input data
        if self.padding == "same":
            # Extract the dimensions of the input data, kernel, and stride
            _, input_height, input_width, _ = x.shape
            kernel_height, kernel_width = self.kernel_size
            stride_height, stride_width = self.stride
            
            # Compute the output shape of the Conv2D layer
            output_height = int(np.ceil(float(input_height) / float(stride_height)))
            output_width = int(np.ceil(float(input_width) / float(stride_width)))
            
            # Compute the padding values along the height and width
            pad_along_height = max((output_height - 1) * stride_height + kernel_height - input_height, 0)
            pad_along_width = max((output_width - 1) * stride_width + kernel_width - input_width, 0)
            
            # Compute the padding values
            padding_top = pad_along_height // 2
            padding_bottom = pad_along_height - padding_top
            padding_left = pad_along_width // 2
            padding_right = pad_along_width - padding_left
            
            # Pad the input data
            x = x.pad((
                (0, 0),
                (padding_top, padding_bottom),
                (padding_left, padding_right),
                (0, 0)
            ))
        
        # Compute the convolution operation by applying the filters to the input data
        out = x.conv_2d(self.filters, stride=self.stride)
        
        # Add the bias to the output
        out = out + self.bias
            
        # Apply the activation function
        if self.activation is not None:
            out = self.activation(out)
        
        # Return the output of the Conv2D layer
        return out
    
    
    def _lazy_init(self, x: Tensor, *args, **kwargs) -> None:
        """
        Function to initialize the filters of the Conv2D layer.
        
        Parameters:
        - x (Tensor): The input Tensor. Shape: (Batch size, Height, Width, Channels)
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
        # Check if the input shape has a valid shape
        assert len(x.shape) == 4, f"Input must be a 4D array. The shape must be (Batch size, Height, Width, Channels). Got shape: {x.shape}"
        
        # Extract the dimensions of the kernel
        kernel_height, kernel_width = self.kernel_size
        num_channels = x.shape[-1]
        
        # Initialize the filters with random values
        self.filters = Tensor(
            data = np.random.randn(self.num_filters, num_channels, kernel_height, kernel_width) / (kernel_height * kernel_width),
            requires_grad = True,
            is_parameter = True
        )
        
        # Initialize the bias with zeros
        self.bias = Tensor(
            data = np.zeros(self.num_filters),
            requires_grad = True,
            is_parameter = True
        )


class ConvTranspose2D(SingleOutputModule):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        num_filters: int, 
        kernel_size: Tuple[int, int], 
        padding: Tuple[int, int] = (0, 0),
        output_padding: Tuple[int, int] = (0, 0),
        stride: Tuple[int, int] = (1, 1), 
        activation: Optional[Activation] = None,
        *args, 
        **kwargs
    ) -> None:
        """
        Class constructor for ConvTranspose2D layer.
        
        Parameters:
        - num_filters (int): Number of output filters (output channels)
        - kernel_size (Tuple[int, int]): Size of the kernel (height, width)
        - padding (Tuple[int, int]): Padding applied to reduce output size. Default is (0, 0)
        - output_padding (Tuple[int, int]): Additional size added to output. Default is (0, 0)
        - stride (Tuple[int, int]): Stride of the transposed convolution. Default is (1, 1)
        - activation (Optional[Activation]): Activation function to apply to output
        
        Raises:
        - AssertionError: if the kernel size is not a tuple of 2 integers
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Check if the kernel size has a valid shape
        assert len(kernel_size) == 2, f"Kernel size must be a tuple of 2 integers. Got: {kernel_size}"
        
        # Initialize the parameters
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.output_padding = output_padding
        self.stride = stride
        self.activation = activation
        
        # Initializing the filters and bias (lazy initialization)
        self.filters: Tensor
        self.bias: Tensor


    ### Protected methods ###
        
    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Function to compute the forward pass of the ConvTranspose2D layer.
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, Height, Width, Channels)
        
        Returns:
        - Tensor: Output data. Shape: (Batch size, Out Height, Out Width, Number of filters)
        """
        
        # Compute the transposed convolution operation
        out = x.conv_transpose_2d(
            kernel = self.filters, 
            stride = self.stride,
            padding = self.padding,
            output_padding = self.output_padding
        )
        
        # Add the bias to the output
        out = out + self.bias
            
        # Apply the activation function
        if self.activation is not None:
            out = self.activation(out)
        
        # Return the output of the ConvTranspose2D layer
        return out
    
    
    def _lazy_init(self, x: Tensor, *args, **kwargs) -> None:
        """
        Function to initialize the filters of the ConvTranspose2D layer.
        
        Parameters:
        - x (Tensor): The input Tensor. Shape: (Batch size, Height, Width, Channels)
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
        # Check if the input shape has a valid shape
        assert len(x.shape) == 4, f"Input must be a 4D array. The shape must be (Batch size, Height, Width, Channels). Got shape: {x.shape}"
        
        # Extract the dimensions of the kernel
        kernel_height, kernel_width = self.kernel_size
        num_channels = x.shape[-1]  # Input channels
        
        # Initialize the filters with random values
        # Shape: (in_channels, out_channels, kernel_height, kernel_width)
        self.filters = Tensor(
            data = np.random.randn(num_channels, self.num_filters, kernel_height, kernel_width) / (kernel_height * kernel_width),
            requires_grad = True,
            is_parameter = True
        )
        
        # Initialize the bias with zeros
        self.bias = Tensor(
            data = np.zeros(self.num_filters),
            requires_grad = True,
            is_parameter = True
        )