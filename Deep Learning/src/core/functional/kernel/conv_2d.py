import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def conv_2d_forward(x_data: np.ndarray, kernel_data: np.ndarray, stride: tuple[int, int]) -> np.ndarray:
    """
    2D Convolution forward pass.
    
    Parameters:
    - x_data (np.ndarray): Input tensor of shape (batch_size, height, width, channels)
    - kernel_data (np.ndarray): Weight tensor of shape (output_channels, channels, kernel_height, kernel_width)
    - stride (tuple[int, int]): Stride for height and width
    
    Returns:
    - np.ndarray: Output tensor of shape (batch_size, output_height, output_width, output_channels)
    
    Raises:
    - AssertionError: If the kernel or stride is too large for the input size.
    - AssertionError: If the number of input channels in the kernel does not match the number of channels in the input tensor.
    """
        
    # Unpack the stride
    stride_h, stride_w = stride
    
    # Extract dimensions of the tensors
    batch_size, height, width, channels = x_data.shape
    output_channels, kernel_in_channels, kernel_height, kernel_width = kernel_data.shape
    
    # Compute output dimensions
    out_height = (height - kernel_height) // stride_h + 1
    out_width = (width - kernel_width) // stride_w + 1
    
    # Check if the dimensions are valid
    assert kernel_in_channels == channels, "w in_channels != x channels"
    assert out_height > 1 and out_width > 1, "Kernel or stride too large for input size"
    
    # Create the output tensor
    out = np.empty((batch_size, out_height, out_width, output_channels), dtype=x_data.dtype)
    
    # Iterate over the batch size
    for b in prange(batch_size):
        # Iterate over the output height
        for i in range(out_height):
            # Compute the height starting index for the current output position
            i0 = i * stride_h
            
            # Iterate over the output width
            for j in range(out_width):
                # Compute the width starting index for the current output position
                j0 = j * stride_w
                
                # Iterate over the output channels
                for c in range(output_channels):
                    # Initialize the accumulator for the convolution operation
                    acc = 0.0
                    
                    # Iterate over the kernel dimensions and channels
                    for ic in range(channels):
                        for di in range(kernel_height):
                            for dj in range(kernel_width):
                                # Cmpute the convolution operation
                                acc += x_data[b, i0+di, j0+dj, ic] * kernel_data[c, ic, di, dj]
              
                    # Store the result in the output tensor
                    out[b, i, j, c] = acc
                    
    # Return the output tensor
    return out


@njit(parallel=True, fastmath=True)
def conv_2d_backward_w(out_grad: np.ndarray, out_buffer: np.ndarray, x_data: np.ndarray,  stride: tuple[int, int]) -> None:
    """
    2D Convolution backward pass for weights.
    
    Parameters:
    - x_data (np.ndarray): Input tensor of shape (batch_size, height, width, channels)
    - out_buffer (np.ndarray): Gradient of the weight tensor of shape (output_channels, channels, kernel_height, kernel_width)
    - out_grad (np.ndarray): Gradient of the output tensor of shape (batch_size, output_height, output_width, output_channels)
    - stride (tuple[int, int]): Stride for height and width
    """
    
    # Unpack the stride
    stride_h, stride_w = stride
    
    # Extract dimensions of the tensors
    batch_size, _, _, channels = x_data.shape
    output_channels, _, kernel_height, kernel_width = out_buffer.shape
    _, out_height, out_width, _ = out_grad.shape

    # Iterate over the output channels
    for oc in prange(output_channels):
        # Iterate over the input channels
        for ic in range(channels):
            # Iterate over the kernel dimensions
            for di in range(kernel_height):
                # Iterate over the kernel width
                for dj in range(kernel_width):
                    # Initialize the accumulator for the gradient of the weights
                    acc = 0.0
                    
                    # Iterate over the batch size
                    for n in range(batch_size):
                        # Iterate over the output height
                        for i in range(out_height):
                            # Compute the height starting index for the current output position
                            i0 = i * stride_h
                            
                            # Iterate over the output width
                            for j in range(out_width):
                                # Compute the width starting index for the current output position
                                j0 = j * stride_w
                                
                                # Compute the gradient of the weights
                                acc += x_data[n, i0+di, j0+dj, ic] * out_grad[n, i, j, oc]
                                
                    # Store the result in the gradient of the weights tensor
                    out_buffer[oc, ic, di, dj] = acc


@njit(parallel=True, fastmath=True)
def conv_2d_backward_x(out_grad: np.ndarray, out_buffer: np.ndarray, kernel_data: np.ndarray, stride: tuple[int, int]) -> None:
    """
    2D Convolution backward pass for input.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output tensor of shape (batch_size, output_height, output_width, output_channels)
    - out_buffer (np.ndarray): Gradient of the input tensor of shape (batch_size, height, width, channels)
    - kernel_data (np.ndarray): Weight tensor of shape (output_channels, channels, kernel_height, kernel_width)
    - stride (tuple[int, int]): Stride for height and width
    """
    
    # Unpack the stride
    stride_h, stride_w = stride
    
    # Extract dimensions of the tensors
    batch_size, output_height, output_width, channels = out_grad.shape
    _, output_channels, kernel_height, kernel_width = kernel_data.shape
    
    # Iterate over the batch size
    for b in prange(batch_size):
        # Iterate over the output height
        for i in range(output_height):
            # Compute the height starting index for the current output position
            i0 = i * stride_h
            
            # Iterate over the output width
            for j in range(output_width):
                # Compute the width starting index for the current output position
                j0 = j * stride_w
                
                # Iterate over the output channels
                for oc in range(channels):
                    # Compute the gradient of the output
                    go = out_grad[b, i, j, oc]
                    
                    # Iterate over the kernel dimensions and channels
                    for ic in range(output_channels):
                        # Iterate over the kernel height
                        for di in range(kernel_height):
                            # Iterate over the kernel width
                            for dj in range(kernel_width):
                                # Compute the gradient of the input
                                out_buffer[b, i0+di, j0+dj, ic] += kernel_data[oc, ic, di, dj] * go