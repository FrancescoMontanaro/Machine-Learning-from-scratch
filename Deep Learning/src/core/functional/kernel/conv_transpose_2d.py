import numpy as np
from numba import njit, prange


############################
#### Internal Functions ####
############################

@njit(fastmath=True)
def _col2im_add(
    col_data: np.ndarray,
    out_buffer: np.ndarray,
    kernel_height: int,
    kernel_width: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    in_height: int,
    in_width: int,
    out_height: int,
    out_width: int,
    out_channels: int
) -> None:
    """
    Scatter-add column data back to image format for a single batch.
    
    Parameters:
    - col_data (np.ndarray): Column data of shape (in_height * in_width, out_channels * kernel_height * kernel_width)
    - out_buffer (np.ndarray): Output buffer of shape (out_height, out_width, out_channels)
    - Other params: convolution parameters
    """

    col_idx = 0
    for i_in in range(in_height):
        for j_in in range(in_width):
            flat_idx = 0
            for c_out in range(out_channels):
                for kh in range(kernel_height):
                    i_out = i_in * stride_h - pad_h + kh
                    for kw in range(kernel_width):
                        j_out = j_in * stride_w - pad_w + kw
                        if 0 <= i_out < out_height and 0 <= j_out < out_width:
                            out_buffer[i_out, j_out, c_out] += col_data[col_idx, flat_idx]
                        flat_idx += 1
            col_idx += 1


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
    Internal function for transposed convolution forward pass using GEMM + col2im.
    
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

    # Reshape kernel for GEMM: (in_channels, out_channels * kH * kW)
    kernel_flat = kernel_data.reshape(in_channels, out_channels * kernel_height * kernel_width)
    
    # Process batches in parallel
    for b in prange(batch_size):
        # Make batch slice contiguous and reshape: (in_height * in_width, in_channels)
        x_batch = np.ascontiguousarray(x_data[b]).reshape(in_height * in_width, in_channels)
        
        # GEMM: (in_height * in_width, in_channels) @ (in_channels, out_channels * kH * kW)
        # Result: (in_height * in_width, out_channels * kH * kW)
        col_data = np.dot(x_batch, kernel_flat)
        
        # Scatter-add to output using col2im
        _col2im_add(
            col_data, out_data[b],
            kernel_height, kernel_width,
            stride_h, stride_w, pad_h, pad_w,
            in_height, in_width, out_height, out_width, out_channels
        )


@njit(fastmath=True)
def _im2col_backward_x(
    out_grad: np.ndarray,
    col_buffer: np.ndarray,
    kernel_height: int,
    kernel_width: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    in_height: int,
    in_width: int,
    out_height: int,
    out_width: int,
    out_channels: int
) -> None:
    """
    Gather gradient patches for backward pass (im2col for gradient).
    
    Parameters:
    - out_grad (np.ndarray): Gradient of output of shape (out_height, out_width, out_channels)
    - col_buffer (np.ndarray): Buffer of shape (in_height * in_width, out_channels * kernel_height * kernel_width)
    """

    col_idx = 0
    for i_in in range(in_height):
        for j_in in range(in_width):
            flat_idx = 0
            for c_out in range(out_channels):
                for kh in range(kernel_height):
                    i_out = i_in * stride_h - pad_h + kh
                    for kw in range(kernel_width):
                        j_out = j_in * stride_w - pad_w + kw
                        if 0 <= i_out < out_height and 0 <= j_out < out_width:
                            col_buffer[col_idx, flat_idx] = out_grad[i_out, j_out, c_out]
                        else:
                            col_buffer[col_idx, flat_idx] = 0.0
                        flat_idx += 1
            col_idx += 1


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
    Backward pass for transposed convolution with respect to input using GEMM.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of output of shape (batch_size, out_height, out_width, out_channels)
    - dx_buffer (np.ndarray): Buffer for input gradient of shape (batch_size, in_height, in_width, in_channels)
    - kernel_data (np.ndarray): Kernel data of shape (in_channels, out_channels, kernel_height, kernel_width)
    - Other parameters: Same as forward
    """
    
    # Reshape kernel for GEMM: (in_channels, out_channels * kH * kW)
    kernel_flat = kernel_data.reshape(in_channels, out_channels * kernel_height * kernel_width)
    
    # Process batches in parallel
    for b in prange(batch_size):
        # Allocate column buffer for this batch
        col_buffer = np.empty((in_height * in_width, out_channels * kernel_height * kernel_width), dtype=out_grad.dtype)
        
        # Gather gradient patches using im2col pattern
        _im2col_backward_x(
            out_grad[b], col_buffer,
            kernel_height, kernel_width,
            stride_h, stride_w, pad_h, pad_w,
            in_height, in_width, out_height, out_width, out_channels
        )
        
        # GEMM: (in_height * in_width, out_channels * kH * kW) @ (out_channels * kH * kW, in_channels)
        # Result: (in_height * in_width, in_channels)
        dx_flat = np.dot(col_buffer, kernel_flat.T)
        
        # Reshape and add to buffer
        dx_buffer[b] += dx_flat.reshape(in_height, in_width, in_channels)


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
    Backward pass for transposed convolution with respect to kernel weights using GEMM.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of output of shape (batch_size, out_height, out_width, out_channels)
    - dw_buffer (np.ndarray): Buffer for kernel gradient of shape (in_channels, out_channels, kernel_height, kernel_width)
    - x_data (np.ndarray): Input data of shape (batch_size, in_height, in_width, in_channels)
    - Other parameters: Same as forward
    """

    # Accumulate gradients per input channel (parallelize over in_channels)
    for c_in in prange(in_channels):
        # Local accumulator for this channel
        local_dw = np.zeros((out_channels, kernel_height, kernel_width), dtype=dw_buffer.dtype)
        
        for b in range(batch_size):
            for i_in in range(in_height):
                for j_in in range(in_width):
                    x_val = x_data[b, i_in, j_in, c_in]
                    
                    # Skip if input is zero (common in sparse inputs)
                    if x_val == 0.0:
                        continue
                    
                    for kh in range(kernel_height):
                        i_out = i_in * stride_h - pad_h + kh
                        if i_out < 0 or i_out >= out_height:
                            continue
                        for kw in range(kernel_width):
                            j_out = j_in * stride_w - pad_w + kw
                            if j_out < 0 or j_out >= out_width:
                                continue
                            for c_out in range(out_channels):
                                local_dw[c_out, kh, kw] += x_val * out_grad[b, i_out, j_out, c_out]
        
        # Write accumulated gradients
        dw_buffer[c_in] += local_dw


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
