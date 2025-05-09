import numpy as np
from numba import njit, prange


############################
#### Internal Functions ####
############################

@njit(parallel=True, fastmath=True)
def img_to_col(
    x_data: np.ndarray, 
    kernel_data: np.ndarray, 
    stride_h: int, 
    stride_w: int,
    batch_size: int, 
    channels: int,
    output_channels: int, 
    kernel_height: int, 
    kernel_width: int,
    out_height: int, 
    out_width: int
) -> np.ndarray:
    """
    Function to convert an image to column (im2col) format for convolution.
    Columns are flattened patches of the image.
    
    Parameters:
    - x_data (np.ndarray): Input data of shape (batch_size, height, width, channels)
    - kernel_data (np.ndarray): Kernel data of shape (output_channels, channels, kernel_height, kernel_width)
    - stride_h (int): Stride for height
    - stride_w (int): Stride for width
    - batch_size (int): Number of batches
    - channels (int): Number of input channels
    - output_channels (int): Number of output channels
    - kernel_height (int): Height of the kernel
    - kernel_width (int): Width of the kernel
    - out_height (int): Output height
    - out_width (int): Output width
    """
    
    # Reshape kernel_data to prepare for GEMM
    kernel_matrix_gemm = kernel_data.reshape(output_channels, channels * kernel_height * kernel_width)
    
    # Initialize the output array
    out_arr = np.empty((batch_size, out_height, out_width, output_channels), dtype=x_data.dtype)

    # Loop over each batch
    for b in prange(batch_size):
        # Extract the current batch of data
        x_item = x_data[b]
        
        # Initialize the patches for the current item
        patches_for_item = np.empty((out_height * out_width, kernel_height * kernel_width * channels), dtype=x_data.dtype)
        
        # Initialize the patch index
        patch_idx = 0
        
        # Assign the patches for the current item
        for i in range(out_height):
            h_start = i * stride_h
            for j in range(out_width):
                w_start = j * stride_w
                current_patch_flat_idx = 0
                for c_idx_k in range(channels):
                    for kh_k in range(kernel_height):
                        for kw_k in range(kernel_width):
                            patches_for_item[patch_idx, current_patch_flat_idx] = x_item[h_start + kh_k, w_start + kw_k, c_idx_k]
                            current_patch_flat_idx += 1
                            
                # Increment the patch index
                patch_idx += 1
        
        # Perform the GEMM operation
        result_gemm = np.dot(patches_for_item, kernel_matrix_gemm.T)
        
        # Reshape the result and assign it to the output array
        out_arr[b] = result_gemm.reshape(out_height, out_width, output_channels)
            
    # Return the output array
    return out_arr


@njit(fastmath=True)
def col_to_img(
    cols: np.ndarray,
    image_buffer: np.ndarray,
    in_channels: int, 
    kernel_height: int, 
    kernel_width: int,
    stride_h: int, 
    stride_w: int,
    output_height: int, 
    output_width: int
) -> None:
    """
    Function to convert columns back to image format.
    
    Parameters:
    - cols (np.ndarray): Input columns of shape (in_channels * kernel_height * kernel_width, output_height * output_width)
    - image_buffer (np.ndarray): Output image buffer of shape (output_height, output_width, in_channels)
    - in_channels (int): Number of input channels
    - kernel_height (int): Height of the kernel
    - kernel_width (int): Width of the kernel
    - stride_h (int): Stride for height
    - stride_w (int): Stride for width
    - output_height (int): Output height for forward pass
    - output_width (int): Output width for forward pass
    """
    
    # Iterate over the output dimensions
    current_col_idx = 0
    for i_fwd_out in range(output_height):
        h_start_img = i_fwd_out * stride_h
        for j_fwd_out in range(output_width):
            # Calculate the starting position in the image buffer
            w_start_img = j_fwd_out * stride_w
            flat_element_idx_in_patch = 0
            
            # Reconstructuring the patch
            for c_k_loop in range(in_channels):
                for kh_k_loop in range(kernel_height):
                    for kw_k_loop in range(kernel_width):
                        val_to_add = cols[flat_element_idx_in_patch, current_col_idx]
                        image_buffer[h_start_img + kh_k_loop, w_start_img + kw_k_loop, c_k_loop] += val_to_add
                        flat_element_idx_in_patch += 1
            
            # Increment the column index
            current_col_idx += 1

     
@njit(parallel=True, fastmath=True)
def build_x_patches(
    x_data: np.ndarray,
    all_patches_buffer: np.ndarray,
    kernel_height: int, 
    kernel_width: int, 
    stride_h: int, 
    stride_w: int,
    in_channels: int, 
    output_height: int, 
    output_width: int, 
    batch_size: int
) -> None:
    """
    Function to build patches from the input data for convolution.
    
    Parameters:
    - x_data (np.ndarray): Input data of shape (batch_size, height, width, channels)
    - all_patches_buffer (np.ndarray): Output buffer for patches of shape (batch_size * output_height * output_width, in_channels * kernel_height * kernel_width)
    - kernel_height (int): Height of the kernel
    - kernel_width (int): Width of the kernel
    - stride_h (int): Stride for height
    - stride_w (int): Stride for width
    - in_channels (int): Number of input channels
    - output_height (int): Output height for forward pass
    - output_width (int): Output width for forward pass
    - batch_size (int): Number of batches
    """
    
    # Iterate over the batches
    for n in prange(batch_size):
        x_item = x_data[n]
        
        # Initialize the starting index for the current batch
        patch_row_start_for_this_batch = n * (output_height * output_width)
        
        # Iterate over the output dimensions
        for i_output_height in range(output_height):
            img_h_start = i_output_height * stride_h
            for i_output_width in range(output_width):
                
                # Calculate the starting position in the image
                img_w_start = i_output_width * stride_w
                current_patch_row_idx = patch_row_start_for_this_batch + (i_output_height * output_width + i_output_width)
                flat_element_idx_in_patch = 0

                # Assign the patch to the buffer
                for c_idx_loop in range(in_channels):
                    for kh_idx_loop in range(kernel_height):
                        for kw_idx_loop in range(kernel_width):
                            all_patches_buffer[current_patch_row_idx, flat_element_idx_in_patch] = \
                                x_item[img_h_start + kh_idx_loop, img_w_start + kw_idx_loop, c_idx_loop]
                                
                            # Increment the flat index for the patch
                            flat_element_idx_in_patch += 1


@njit(fastmath=True)
def conv_2d_backward_w_internal(
    out_grad: np.ndarray,
    dw_buffer: np.ndarray,
    x_data: np.ndarray,
    stride_h: int,
    stride_w: int,
    batch_size: int,
    in_channels: int,
    out_channels: int, 
    kernel_height: int, 
    kernel_width: int,
    output_height: int, 
    output_width: int
) -> None:
    """
    Function to compute the gradient of the weights for 2D convolution.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output data of shape (batch_size, out_height, out_width, output_channels)
    - dw_buffer (np.ndarray): Buffer for the weights of shape (output_channels, input_channels, kernel_height, kernel_width)
    - x_data (np.ndarray): Input data of shape (batch_size, height, width, input_channels)
    - stride_h (int): Stride for height
    - stride_w (int): Stride for width
    - batch_size (int): Number of batches
    - in_channels (int): Number of input channels
    - out_channels (int): Number of output channels
    - kernel_height (int): Height of the kernel
    - kernel_width (int): Width of the kernel
    - output_height (int): Output height for forward pass
    - output_width (int): Output width for forward pass
    """

    # Create a temporary buffer for the patches
    x_patches_all = np.empty((batch_size * output_height * output_width, in_channels * kernel_height * kernel_width), dtype=x_data.dtype)
    
    # Build the patches from the input data
    build_x_patches(
        x_data, x_patches_all, kernel_height, kernel_width, stride_h, stride_w,
        in_channels, output_height, output_width, batch_size
    )

    # Reshape the output gradient
    dy_flat = out_grad.reshape(batch_size * output_height * output_width, out_channels)
    
    # Perform the GEMM operation
    dw_temp_flat = np.dot(x_patches_all.T, dy_flat)

    # Reshape the result to match the kernel dimensions
    dw_reshaped_intermediate = dw_temp_flat.reshape(in_channels, kernel_height, kernel_width, out_channels)
    
    # Transpose the result to match the expected output shape
    final_dw_values = dw_reshaped_intermediate.transpose(3, 0, 1, 2)
    
    # Assign the result to the output buffer
    dw_buffer[:] = final_dw_values             
                    
                    
@njit(parallel=True, fastmath=True)
def conv_2d_backward_x_internal(
    out_grad: np.ndarray,
    dx_buffer: np.ndarray,
    kernel_data: np.ndarray,
    stride_h: int, 
    stride_w: int,
    batch_size: int,
    in_channels: int,
    out_channels: int, 
    kernel_height: int, 
    kernel_width: int,
    output_height: int,
    output_width: int
) -> None:
    """
    Function to compute the gradient of the input for 2D convolution.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output data of shape (batch_size, out_height, out_width, output_channels)
    - dx_buffer (np.ndarray): Buffer for the input data of shape (batch_size, height, width, input_channels)
    - kernel_data (np.ndarray): Kernel data of shape (output_channels, input_channels, kernel_height, kernel_width)
    - stride_h (int): Stride for height
    - stride_w (int): Stride for width
    - batch_size (int): Number of batches
    - in_channels (int): Number of input channels
    - out_channels (int): Number of output channels
    - kernel_height (int): Height of the kernel
    - kernel_width (int): Width of the kernel
    - output_height (int): Output height for forward pass
    - output_width (int): Output width for forward pass
    """

    # Reshape the kernel data for GEMM
    kernel_fwd_matrix = kernel_data.reshape(out_channels, in_channels * kernel_height * kernel_width)
    
    # Transpose the kernel matrix for GEMM
    w_t_for_gemm = kernel_fwd_matrix.T # Shape: (in_channels * kernel_height * kernel_width, out_channels)

    # Iterate over the batches
    for b_idx in prange(batch_size):
        # Slice the output gradient for the current batch
        dy_item_sliced = out_grad[b_idx] # Shape (output_height, output_width, out_channels)
        
        # Transpose the sliced output gradient to prepare for GEMM
        dy_item_transposed = dy_item_sliced.transpose(2, 0, 1)
        dy_item_transposed_contiguous = dy_item_transposed.copy()
        
        # Reshape the transposed output gradient for GEMM
        dy_item_flat_for_gemm = dy_item_transposed_contiguous.reshape(out_channels, output_height * output_width) # Shape: (out_channels, output_height * output_width)

        # Perform the GEMM operation
        dx_cols_item = np.dot(w_t_for_gemm, dy_item_flat_for_gemm) # Shape: (in_channels*kernel_height*kernel_width, output_height*output_width)

        # Call the function to convert columns back to image format
        col_to_img(
            cols = dx_cols_item, 
            image_buffer = dx_buffer[b_idx], 
            in_channels = in_channels, 
            kernel_height = kernel_height, 
            kernel_width = kernel_width,
            stride_h = stride_h,
            stride_w = stride_w, 
            output_height = output_height,
            output_width = output_width
        )


#############################
#### Interface Functions ####
#############################

def conv_2d_forward(x_data: np.ndarray, kernel_data: np.ndarray, stride: tuple[int, int]) -> np.ndarray:
    """
    Forward pass for 2D convolution
    
    Parameters:
    - x_data (np.ndarray): Input data of shape (batch_size, height, width, channels)
    - kernel_data (np.ndarray): Kernel data of shape (output_channels, channels, kernel_height, kernel_width)
    - stride (tuple[int, int]): Stride for the convolution
    
    Returns:
    - np.ndarray: Output data of shape (batch_size, out_height, out_width, output_channels)
    """
    
    # Unpack the input data
    stride_h, stride_w = stride
    batch_size, height, width, channels = x_data.shape
    output_channels, _, kernel_height, kernel_width = kernel_data.shape
    
    # Compute the output dimensions
    out_height = (height - kernel_height) // stride_h + 1
    out_width = (width - kernel_width) // stride_w + 1
    
    # Assert the output dimensions are valid
    assert out_height > 0 and out_width > 0, "Output dimensions must be positive"
    assert kernel_data.shape[1] == channels, "Input channels must match kernel channels"

    # Call the internal function to perform the convolution
    return img_to_col(
        x_data = x_data,
        kernel_data = kernel_data,
        stride_h = stride_h,
        stride_w = stride_w,
        batch_size = batch_size,
        channels = channels,
        output_channels = output_channels,
        kernel_height = kernel_height,
        kernel_width = kernel_width,
        out_height = out_height,
        out_width = out_width
    )


def conv_2d_backward_w(out_grad: np.ndarray, out_buffer: np.ndarray, x_data: np.ndarray, stride: tuple[int, int]) -> None:
    """
    Backward pass for the weights of 2D convolution
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output data of shape (batch_size, out_height, out_width, output_channels)
    - out_buffer (np.ndarray): Buffer for the weights of shape (output_channels, input_channels, kernel_height, kernel_width)
    - x_data (np.ndarray): Input data of shape (batch_size, height, width, input_channels)
    - stride (tuple[int, int]): Stride for the convolution
    """
    
    # Unpack the input data
    stride_h, stride_w = stride
    batch_size, _, _, channels = x_data.shape
    output_channels, _, kernel_height, kernel_width = out_buffer.shape
    _, output_height, output_width, _ = out_grad.shape
    
    # Call the internal function to perform the backward pass for weights
    conv_2d_backward_w_internal(
        out_grad = out_grad,
        dw_buffer = out_buffer,
        x_data = x_data,
        stride_h = stride_h,
        stride_w = stride_w,
        batch_size = batch_size,
        in_channels = channels,
        out_channels = output_channels,
        kernel_height = kernel_height,
        kernel_width = kernel_width,
        output_height = output_height,
        output_width = output_width
    )  


def conv_2d_backward_x(out_grad: np.ndarray, out_buffer: np.ndarray, kernel_data: np.ndarray, stride: tuple[int, int]) -> None:
    """
    Backward pass for the input of 2D convolution
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output data of shape (batch_size, out_height, out_width, output_channels)
    - out_buffer (np.ndarray): Buffer for the input data of shape (batch_size, height, width, input_channels)
    - kernel_data (np.ndarray): Kernel data of shape (output_channels, input_channels, kernel_height, kernel_width)
    - stride (tuple[int, int]): Stride for the convolution
    """
    
    # Unpack the input data
    stride_h, stride_w = stride
    batch_size, output_height, output_width, out_channels_from_og = out_grad.shape
    out_channels, in_channels, kernel_height, kernel_width = kernel_data.shape
    
    # Assert the output dimensions are valid
    assert out_channels_from_og == out_channels, "Input channels must match output channels"

    # Call the internal function to perform the backward pass for input
    conv_2d_backward_x_internal(
        out_grad = out_grad,
        dx_buffer = out_buffer,
        kernel_data = kernel_data,
        stride_h = stride_h,
        stride_w = stride_w,
        batch_size = batch_size,
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_height = kernel_height,
        kernel_width = kernel_width,
        output_height = output_height,
        output_width = output_width
    )