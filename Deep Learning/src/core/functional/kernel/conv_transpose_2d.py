import numpy as np
from numba import njit, prange


############################
#### Internal Functions ####
############################

@njit(parallel=True, fastmath=True)
def conv_transpose_2d_forward_internal(
    x_data: np.ndarray,
    kernel_data: np.ndarray,
    out_data: np.ndarray,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    kernel_height: int,
    kernel_width: int,
    in_height: int,
    in_width: int,
    out_height: int,
    out_width: int
) -> None:
    """
    Internal function for transposed convolution forward pass.
    
    Parameters:
    - x_data (np.ndarray): Input data of shape (batch_size, in_height, in_width, in_channels)
    - kernel_data (np.ndarray): Kernel data of shape (in_channels, out_channels, kernel_height, kernel_width)
    - out_data (np.ndarray): Output buffer of shape (batch_size, out_height, out_width, out_channels)
    - stride_h, stride_w (int): Stride values
    - pad_h, pad_w (int): Padding values
    - batch_size, in_channels, out_channels (int): Dimension sizes
    - kernel_height, kernel_width (int): Kernel dimensions
    - in_height, in_width (int): Input spatial dimensions
    - out_height, out_width (int): Output spatial dimensions
    """
    
    # Iterate over batches in parallel
    for b in prange(batch_size):
        # Iterate over input spatial positions
        for i_in in range(in_height):
            for j_in in range(in_width):
                # Iterate over input channels
                for c_in in range(in_channels):
                    # Get input value
                    x_val = x_data[b, i_in, j_in, c_in]
                    
                    # Iterate over kernel positions
                    for kh in range(kernel_height):
                        for kw in range(kernel_width):
                            # Calculate output position
                            i_out = i_in * stride_h - pad_h + kh
                            j_out = j_in * stride_w - pad_w + kw
                            
                            # Check bounds
                            if 0 <= i_out < out_height and 0 <= j_out < out_width:
                                # Iterate over output channels
                                for c_out in range(out_channels):
                                    # Kernel layout: (in_channels, out_channels, kH, kW)
                                    out_data[b, i_out, j_out, c_out] += x_val * kernel_data[c_in, c_out, kh, kw]


@njit(parallel=True, fastmath=True)
def conv_transpose_2d_backward_x_internal(
    out_grad: np.ndarray,
    dx_buffer: np.ndarray,
    kernel_data: np.ndarray,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    kernel_height: int,
    kernel_width: int,
    in_height: int,
    in_width: int,
    out_height: int,
    out_width: int
) -> None:
    """
    Backward pass for transposed convolution with respect to input.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of output of shape (batch_size, out_height, out_width, out_channels)
    - dx_buffer (np.ndarray): Buffer for input gradient of shape (batch_size, in_height, in_width, in_channels)
    - kernel_data (np.ndarray): Kernel data of shape (in_channels, out_channels, kernel_height, kernel_width)
    - Other parameters: Same as forward
    """
    
    # Iterate over batches in parallel
    for b in prange(batch_size):
        # Iterate over input spatial positions
        for i_in in range(in_height):
            for j_in in range(in_width):
                # Iterate over input channels
                for c_in in range(in_channels):
                    grad_sum = 0.0
                    
                    # Iterate over kernel positions
                    for kh in range(kernel_height):
                        for kw in range(kernel_width):
                            # Calculate output position
                            i_out = i_in * stride_h - pad_h + kh
                            j_out = j_in * stride_w - pad_w + kw
                            
                            # Check bounds
                            if 0 <= i_out < out_height and 0 <= j_out < out_width:
                                # Sum contributions from all output channels
                                for c_out in range(out_channels):
                                    grad_sum += out_grad[b, i_out, j_out, c_out] * kernel_data[c_in, c_out, kh, kw]
                    
                    dx_buffer[b, i_in, j_in, c_in] += grad_sum


@njit(parallel=True, fastmath=True)
def conv_transpose_2d_backward_w_internal(
    out_grad: np.ndarray,
    dw_buffer: np.ndarray,
    x_data: np.ndarray,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    kernel_height: int,
    kernel_width: int,
    in_height: int,
    in_width: int,
    out_height: int,
    out_width: int
) -> None:
    """
    Backward pass for transposed convolution with respect to kernel weights.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of output of shape (batch_size, out_height, out_width, out_channels)
    - dw_buffer (np.ndarray): Buffer for kernel gradient of shape (in_channels, out_channels, kernel_height, kernel_width)
    - x_data (np.ndarray): Input data of shape (batch_size, in_height, in_width, in_channels)
    - Other parameters: Same as forward
    """
    
    # Iterate over kernel positions and channels in parallel
    for c_in in prange(in_channels):
        for c_out in range(out_channels):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    grad_sum = 0.0
                    
                    # Sum over batches and spatial positions
                    for b in range(batch_size):
                        for i_in in range(in_height):
                            for j_in in range(in_width):
                                # Calculate output position
                                i_out = i_in * stride_h - pad_h + kh
                                j_out = j_in * stride_w - pad_w + kw
                                
                                # Check bounds
                                if 0 <= i_out < out_height and 0 <= j_out < out_width:
                                    grad_sum += x_data[b, i_in, j_in, c_in] * out_grad[b, i_out, j_out, c_out]
                    
                    dw_buffer[c_in, c_out, kh, kw] += grad_sum


#############################
#### Interface Functions ####
#############################

def conv_transpose_2d_forward(
    x_data: np.ndarray, 
    kernel_data: np.ndarray, 
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    output_padding: tuple[int, int] = (0, 0)
) -> np.ndarray:
    """
    Forward pass for 2D transposed convolution (deconvolution).
    
    Parameters:
    - x_data (np.ndarray): Input data of shape (batch_size, height, width, in_channels)
    - kernel_data (np.ndarray): Kernel data of shape (in_channels, out_channels, kernel_height, kernel_width)
    - stride (tuple[int, int]): Stride for the transposed convolution
    - padding (tuple[int, int]): Padding to remove from output
    - output_padding (tuple[int, int]): Additional padding to add to output
    
    Returns:
    - np.ndarray: Output data of shape (batch_size, out_height, out_width, out_channels)
    """
    
    # Unpack parameters
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    out_pad_h, out_pad_w = output_padding
    
    # Extract dimensions
    batch_size, in_height, in_width, in_channels = x_data.shape
    in_channels_k, out_channels, kernel_height, kernel_width = kernel_data.shape
    
    # Validate dimensions
    assert in_channels == in_channels_k, f"Input channels mismatch: {in_channels} vs {in_channels_k}"
    
    # Compute output dimensions
    # Formula: out = (in - 1) * stride - 2 * padding + kernel_size + output_padding
    out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_height + out_pad_h
    out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_width + out_pad_w
    
    assert out_height > 0 and out_width > 0, f"Invalid output dimensions: {out_height}x{out_width}"
    
    # Initialize output buffer
    out_data = np.zeros((batch_size, out_height, out_width, out_channels), dtype=x_data.dtype)
    
    # Perform transposed convolution
    conv_transpose_2d_forward_internal(
        x_data = x_data,
        kernel_data = kernel_data,
        out_data = out_data,
        stride_h = stride_h,
        stride_w = stride_w,
        pad_h = pad_h,
        pad_w = pad_w,
        batch_size = batch_size,
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_height = kernel_height,
        kernel_width = kernel_width,
        in_height = in_height,
        in_width = in_width,
        out_height = out_height,
        out_width = out_width
    )
    
    return out_data


def conv_transpose_2d_backward_x(
    out_grad: np.ndarray,
    out_buffer: np.ndarray,
    kernel_data: np.ndarray,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    output_padding: tuple[int, int] = (0, 0)
) -> None:
    """
    Backward pass for transposed convolution with respect to input.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of output of shape (batch_size, out_height, out_width, out_channels)
    - out_buffer (np.ndarray): Buffer for input gradient of shape (batch_size, in_height, in_width, in_channels)
    - kernel_data (np.ndarray): Kernel data of shape (in_channels, out_channels, kernel_height, kernel_width)
    - stride, padding, output_padding: Same as forward
    """
    
    # Unpack parameters
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    # Extract dimensions
    batch_size, in_height, in_width, in_channels = out_buffer.shape
    _, out_channels, kernel_height, kernel_width = kernel_data.shape
    _, out_height, out_width, _ = out_grad.shape
    
    # Compute backward
    conv_transpose_2d_backward_x_internal(
        out_grad = out_grad,
        dx_buffer = out_buffer,
        kernel_data = kernel_data,
        stride_h = stride_h,
        stride_w = stride_w,
        pad_h = pad_h,
        pad_w = pad_w,
        batch_size = batch_size,
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_height = kernel_height,
        kernel_width = kernel_width,
        in_height = in_height,
        in_width = in_width,
        out_height = out_height,
        out_width = out_width
    )


def conv_transpose_2d_backward_w(
    out_grad: np.ndarray,
    out_buffer: np.ndarray,
    x_data: np.ndarray,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    output_padding: tuple[int, int] = (0, 0)
) -> None:
    """
    Backward pass for transposed convolution with respect to kernel weights.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of output of shape (batch_size, out_height, out_width, out_channels)
    - out_buffer (np.ndarray): Buffer for kernel gradient of shape (in_channels, out_channels, kernel_height, kernel_width)
    - x_data (np.ndarray): Input data of shape (batch_size, in_height, in_width, in_channels)
    - stride, padding, output_padding: Same as forward
    """
    
    # Unpack parameters
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    # Extract dimensions
    batch_size, in_height, in_width, in_channels = x_data.shape
    _, out_channels, kernel_height, kernel_width = out_buffer.shape
    _, out_height, out_width, _ = out_grad.shape
    
    # Compute backward
    conv_transpose_2d_backward_w_internal(
        out_grad = out_grad,
        dw_buffer = out_buffer,
        x_data = x_data,
        stride_h = stride_h,
        stride_w = stride_w,
        pad_h = pad_h,
        pad_w = pad_w,
        batch_size = batch_size,
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_height = kernel_height,
        kernel_width = kernel_width,
        in_height = in_height,
        in_width = in_width,
        out_height = out_height,
        out_width = out_width
    )
