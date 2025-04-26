from numba import njit, prange


@njit(parallel=True, fastmath=True)
def max_pool_2d_forward(x, kernel_height, kernel_width, stride_height, stride_width, out, argmax_i, argmax_j) -> None:
    """
    2D Max Pooling forward pass.
    
    Parameters:
    - x (np.ndarray): Input tensor of shape (batch_size, height, width, channels)
    - kernel_height (int): Height of the pooling kernel
    - kernel_width (int): Width of the pooling kernel
    - stride_height (int): Stride height
    - stride_width (int): Stride width
    - out (np.ndarray): Output tensor of shape (batch_size, output_height, output_width, channels)
    - argmax_i (np.ndarray): Indices of the maximum values in the height dimension
    - argmax_j (np.ndarray): Indices of the maximum values in the width dimension
    """
    
    # Extract dimensions of the tensors
    batch_size, _, _, channels = x.shape
    output_height, output_width = out.shape[1], out.shape[2]

    # Iterate over the batch size
    for n in prange(batch_size):
        # Iterate over the output height
        for i in range(output_height):
            # Compute the height starting index for the current output position
            i0 = i * stride_height
            
            # Iterate over the output width
            for j in range(output_width):
                # Compute the width starting index for the current output position
                j0 = j * stride_width
                
                # Iterate over the channels
                for c in range(channels):
                    
                    # Initialize the maximum value and its indices
                    best = -1e30
                    bi = bj = 0
                    
                    # Iterate over the kernel dimensions
                    for di in range(kernel_height):
                        # Iterate over the kernel width
                        for dj in range(kernel_width):
                            # Compute the value at the current position
                            v = x[n, i0+di, j0+dj, c]
                            
                            # Check if the current value is greater than the best found so far
                            if v > best:
                                # Update the best value and its indices
                                best = v
                                
                                # Update the indices of the maximum value
                                bi, bj = di, dj
                                
                    # Store the result in the output tensor
                    out[n, i, j, c] = best
        
                    # Store the indices of the maximum value
                    argmax_i[n, i, j, c] = bi
                    argmax_j[n, i, j, c] = bj
                    
                    
@njit(parallel=True, fastmath=True)
def max_pool_2d_gradient(arg_i, arg_j, grad_out, stride_height, stride_width, grad_x):
    """
    2D Max Pooling gradient
    
    Parameters:
    - arg_i (np.ndarray): Indices of the maximum values in the height dimension
    - arg_j (np.ndarray): Indices of the maximum values in the width dimension
    - og (np.ndarray): Gradient of the output tensor of shape (batch_size, output_height, output_width, channels)
    - stride_height (int): Stride height
    - stride_width (int): Stride width
    - dx (np.ndarray): Gradient of the input tensor of shape (batch_size, height, width, channels)
    """
    
    # Extract dimensions of the tensors
    batch_size, output_height, output_width, channels = grad_out.shape
    
    # Iterate over the batch size
    for n in prange(batch_size):
        # Iterate over the output height
        for i in range(output_height):
            # Compute the height starting index for the current output position
            i0 = i*stride_height
            
            # Iterate over the output width
            for j in range(output_width):
                # Compute the width starting index for the current output position
                j0 = j*stride_width
                
                # Iterate over the channels
                for c in range(channels):
                    # Get the indices of the maximum value
                    di = arg_i[n,i,j,c]
                    dj = arg_j[n,i,j,c]
                    
                    # Compute the gradient of the input
                    grad_x[n, i0+di, j0+dj, c] += grad_out[n,i,j,c]