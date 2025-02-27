import numpy as np
from typing import Literal, Optional

from .base import Layer
from ..optimizers import Optimizer
from ..activations import Activation


class Conv2D(Layer):
    
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
        self.filters = None
        self.bias = None
        
        # Initializing the input shape
        self.input_shape = None
    
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Function to compute the forward pass of the Conv2D layer.
        It initializes the filters if not initialized and computes the forward pass.
        
        Parameters:
        - x (np.ndarray): input data. Shape: (Batch size, Height, Width, Channels)
        
        Returns:
        - np.ndarray: output data. Shape: (Batch size, Height, Width, Number of filters)
        
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
    
    def parameters(self) -> dict:
        """
        Function to return the parameters of the Conv2D layer.
        
        Returns:
        - dict: dictionary containing the filters of the Conv2D layer
        """
        
        return {
            "filters": self.filters
        }
    
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Function to compute the forward pass of the Conv2D layer.
        
        Parameters:
        - x (np.ndarray): input data. Shape: (Batch size, Height, Width, Channels)
        
        Returns:
        - np.ndarray: output data. Shape: (Batch size, Height, Width, Number of filters)
        
        Raises:
        - AssertionError: if the filters are not initialized
        """
        
        # Assert that the filters are initialized
        assert self.filters is not None, "Filters are not initialized. Please call the layer with some input data to initialize the filters."
        assert self.bias is not None, "Bias is not initialized. Please call the layer with some input data to initialize the bias."
        
        # Extract the required dimensions for a better interpretation
        kernel_height, kernel_width = self.kernel_size # Shape of the kernel
        stride_height, stride_width = self.stride # Stride of the kernel
        
        # Apply padding to the input data
        if self.padding == "same":
            # Pad the input data
            x = np.pad(
                x,
                (
                    (0, 0),
                    (self.padding_top, self.padding_bottom),
                    (self.padding_left, self.padding_right),
                    (0, 0)
                ),
                mode="constant"
            )
        
        # Save the input data
        self.x = x
                
        # Extract the patches from the input data
        self.patches = np.lib.stride_tricks.sliding_window_view(
            x,
            window_shape = (kernel_height, kernel_width),
            axis = (1, 2) # type: ignore
        )
        
        # Apply the stride
        self.patches = self.patches[:, ::stride_height, ::stride_width, :, :, :]
        
        # Transpose the patches to have shape: (batch_size, output_height, output_width, kernel_height, kernel_width, num_channels)
        self.patches = self.patches.transpose(0, 1, 2, 4, 5, 3)

        # Compute the output of the Conv2D layer by applying the convolution operation
        output = np.tensordot(self.patches, self.filters, axes=([3, 4, 5], [1, 2, 3])) + self.bias
                
        # Save the feature maps for backpropagation
        self.feature_maps = output        
            
        # Apply the activation function
        if self.activation is not None:
            output = self.activation(self.feature_maps)
        
        return output
    
    
    def backward(self, loss_gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass of the layer (layer i)
        
        Parameters:
        - loss_gradient (np.ndarray): Gradient of the loss with respect to the output of the layer: dL/dO_i
        
        Returns:
        - np.ndarray: Gradient of the loss with respect to the input of the layer: dL/dX_i ≡ dL/dO_{i-1}
        
        Raises:
        - AssertionError: If the weights and bias are not initialized
        """
        
        # Assert that filters and bias are initialized
        assert isinstance(self.optimizer, Optimizer), "Optimizer is not set. Please set an optimizer before training the model."
        assert self.input_shape is not None, "Input shape is not set. Please call the layer with some input data to set the input shape."
        assert self.filters is not None, "Filters are not initialized."
        assert self.bias is not None, "Bias is not initialized."
        
        if self.activation is not None:
            # Multiply the incoming gradient by the activation derivative
            gradient = loss_gradient * self.activation.derivative(self.feature_maps) # dL/dO_i * dO_i/dX_i, where dO_i/dX_i is the activation derivative
        else:
            # If the activation function is not defined, the gradient is the same as the incoming gradient
            gradient = loss_gradient # dL/dO_i
        
        # Extract dimensions
        batch_size, input_height, input_width, num_channels = self.x.shape
        _, output_height, output_width, _ = gradient.shape
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        
        # Create an empty array to store the gradients with respect to the input
        grad_input = np.zeros_like(self.x)
        
        # Compute the gradients with respect to the bias
        grad_bias = np.sum(gradient, axis=(0, 1, 2)) # dL/db_i
        
        # Compute the gradients with respect to the input patches
        grad_input_patches = np.tensordot(gradient, self.filters, axes=([3], [0]))
        
        # Extract the indices of the output patches and flatten them
        out_i, out_j = np.meshgrid(np.arange(output_height), np.arange(output_width), indexing='ij')
        out_i, out_j = out_i.ravel(), out_j.ravel()
        
        # Extract the indices of the kernel
        k_i = np.repeat(np.arange(kernel_height), kernel_width)
        k_j = np.tile(np.arange(kernel_width), kernel_height)
        
        # Compute the indices of the input patches
        rows = (out_i[:, None] * stride_height + k_i[None, :]).ravel()
        cols = (out_j[:, None] * stride_width  + k_j[None, :]).ravel()
        
        # Compute the global indices
        flat_idx = rows * input_width + cols
        
        # Compute the batch offsets
        batch_offsets = np.arange(batch_size) * (input_height * input_width) 
        
        # Compute the global indices
        global_idx = (batch_offsets[:, None] + flat_idx[None, :]).ravel()
        
        # Flatten the gradient patches
        grad_patches_flat = grad_input_patches.reshape(batch_size, -1, num_channels).reshape(-1, num_channels)
        
        # Compute the gradient with respect to the input and reshape it
        grad_input_flat = grad_input.reshape(-1, num_channels)
        np.add.at(grad_input_flat, global_idx, grad_patches_flat)

        # Reshape the gradient with respect to the input
        grad_input = grad_input_flat.reshape(batch_size, input_height, input_width, num_channels)
        
        # Compute the gradient with respect to the filters
        grad_filters = np.tensordot(self.patches, gradient, axes=([0,1,2], [0,1,2]))
        grad_filters = grad_filters.transpose(3, 0, 1, 2)
    
        # Update the filters
        self.filters = self.optimizer.update(
            layer = self,
            param_name = "filters",
            params = self.filters,
            grad_params = grad_filters
        )
        
        # Update bias
        self.bias = self.optimizer.update(
            layer = self,
            param_name = "bias",
            params = self.bias,
            grad_params = grad_bias
        )
        
        # Remove padding if applied during the forward pass
        if self.padding == "same":
            grad_input = grad_input[
                :,
                self.padding_top:-self.padding_bottom if self.padding_bottom != 0 else None,
                self.padding_left:-self.padding_right if self.padding_right != 0 else None,
                :
            ]
        
        return grad_input # dL/dX_i ≡ dL/dO_{i-1}
    
    
    def output_shape(self) -> tuple:
        """
        Function to compute the output shape of the Conv2D layer.
        
        Returns:
        - tuple: shape of the output data
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
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
    
    
    def count_params(self) -> int:
        """
        Function to count the number of parameters in the Conv2D layer.
        
        Returns:
        - int: number of parameters in the Conv2D layer
        
        Raises:
        - AssertionError: if the filters or bias are not initialized
        """
        
        # Assert that the filters are initialized
        assert self.filters is not None, "Filters are not initialized. Please call the layer with some input data to initialize the filters."
        assert self.bias is not None, "Bias is not initialized. Please call the layer with some input data to initialize the bias."
        
        return int(np.prod(self.filters.shape)) + int(np.prod(self.bias.shape)) # num_filters * kernel_size[0] * kernel_size[1] * num_channels + num_filters
    
    
    def init_params(self, num_channels: int) -> None:
        """
        Function to initialize the filters of the Conv2D layer.
        
        Parameters:
        - num_channels (int): number of channels in the input data
        """
        
        # Extract the dimensions of the kernel
        kernel_height, kernel_width = self.kernel_size
        
        # Initializing the filters and bias with random values and normalizing them
        self.filters = np.random.randn(self.num_filters, kernel_height, kernel_width, num_channels) / (kernel_height * kernel_width)
        self.bias = np.zeros(self.num_filters)
        
        # Update the initialization flag
        self.initialized = True