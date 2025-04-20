from numba import njit, prange


@njit(parallel=True)
def pad_forward(x, pad_top, pad_bottom, pad_left, pad_right, out) -> None:
    """
    2D Padding forward pass.
    
    Parameters:
    - x (np.ndarray): Input tensor of shape (batch_size, height, width, channels)
    - pad_top (int): Number of rows to pad at the top
    - pad_bottom (int): Number of rows to pad at the bottom
    - pad_left (int): Number of columns to pad at the left
    - pad_right (int): Number of columns to pad at the right
    - out (np.ndarray): Output tensor of shape (batch_size, height + pad_top + pad_bottom, width + pad_left + pad_right, channels)
    """
    
    # Extract dimensions of the tensors
    batch_size, height, width, channels = x.shape
    output_height, output_width = out.shape[1], out.shape[2]

    # Iterate over the batch size
    for n in prange(batch_size):
        # Iterate over the output height
        for i in range(output_height):
            # Iterate over the output width
            for j in range(output_width):
                # Iterate over the channels
                for c in range(channels):
                    # Compute the corresponding indices in the input tensor
                    ii = i - pad_top
                    jj = j - pad_left
                    
                    # Check if the indices are within bounds
                    if 0 <= ii < height and 0 <= jj < width:
                        # Store the value in the output tensor
                        out[n, i, j, c] = x[n, ii, jj, c]
                    else:
                        # If the indices are out of bounds, set the value to 0
                        out[n, i, j, c] = 0.0


@njit(parallel=True)
def pad_gradient(out_grad, pad_top, pad_bottom, pad_left, pad_right, x_grad) -> None:
    """
    2D Padding gradient
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output tensor of shape (batch_size, output_height, output_width, channels)
    - pad_top (int): Number of rows padded at the top
    - pad_bottom (int): Number of rows padded at the bottom
    - pad_left (int): Number of columns padded at the left
    - pad_right (int): Number of columns padded at the right
    - x_grad (np.ndarray): Gradient of the input tensor of shape (batch_size, height, width, channels)
    """
    
    # Extract dimensions of the tensors
    batch_size, height, width, channels = x_grad.shape
    output_height, output_width = out_grad.shape[1], out_grad.shape[2]
    
    # Iterate over the batch size
    for n in prange(batch_size):
        # Iterate over the output height
        for i in range(output_height):
            # Iterate over the output width
            for j in range(output_width):
                # Iterate over the channels
                for c in range(channels):
                    # Compute the corresponding indices in the input tensor
                    ii = i - pad_top
                    jj = j - pad_left
                    
                    # Check if the indices are within bounds
                    if 0 <= ii < height and 0 <= jj < width:
                        # Accumulate the gradient in the input tensor
                        x_grad[n, ii, jj, c] += out_grad[n, i, j, c]