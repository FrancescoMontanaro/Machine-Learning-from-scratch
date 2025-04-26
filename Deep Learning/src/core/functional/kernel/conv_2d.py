from numba import njit, prange


@njit(parallel=True, fastmath=True)
def conv_2d_forward(x, w, stride_h, stride_w, out) -> None:
    """
    2D Convolution forward pass.
    
    Parameters:
    - x (np.ndarray): Input tensor of shape (batch_size, height, width, channels)
    - w (np.ndarray): Weight tensor of shape (output_channels, channels, kernel_height, kernel_width)
    - stride_h (int): Stride height
    - stride_w (int): Stride width
    """
    
    # Extract dimensions of the tensors
    batch_size, height, width, channels = x.shape
    output_channels, _, kernel_height, kernel_width = w.shape
    
    # Compute output dimensions
    out_height = (height - kernel_height) // stride_h + 1
    out_width = (width - kernel_width) // stride_w + 1
    
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
                                acc += x[b, i0+di, j0+dj, ic] * w[c, ic, di, dj]
              
                    # Store the result in the output tensor
                    out[b, i, j, c] = acc


@njit(parallel=True, fastmath=True)
def conv_2d_gradient_w(x, grad_out, stride_h, stride_w, grad_w) -> None:
    """
    2D Convolution gradient for weights.
    
    Parameters:
    - x (np.ndarray): Input tensor of shape (batch_size, height, width, channels)
    - grad_out (np.ndarray): Gradient of the output tensor of shape (batch_size, output_height, output_width, output_channels)
    - stride_h (int): Stride height
    - stride_w (int): Stride width
    - grad_w (np.ndarray): Gradient of the weight tensor of shape (output_channels, channels, kernel_height, kernel_width)
    """
    
    # Extract dimensions of the tensors
    batch_size, _, _, channels = x.shape
    output_channels, _, kernel_height, kernel_width = grad_w.shape
    _, out_height, out_width, _ = grad_out.shape

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
                                acc += x[n, i0+di, j0+dj, ic] * grad_out[n, i, j, oc]
                                
                    # Store the result in the gradient of the weights tensor
                    grad_w[oc, ic, di, dj] = acc


@njit(parallel=True, fastmath=True)
def conv_2d_gradient_x(grad_out, w, stride_h, stride_w, grad_x) -> None:
    """
    2D Convolution gradient for input.
    
    Parameters:
    - grad_out (np.ndarray): Gradient of the output tensor of shape (batch_size, output_height, output_width, output_channels)
    - w (np.ndarray): Weight tensor of shape (output_channels, channels, kernel_height, kernel_width)
    - stride_h (int): Stride height
    - stride_w (int): Stride width
    - grad_x (np.ndarray): Gradient of the input tensor of shape (batch_size, height, width, channels)
    """
    
    # Extract dimensions of the tensors
    batch_size, output_height, output_width, channels = grad_out.shape
    _, output_channels, kernel_height, kernel_width = w.shape
    
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
                    go = grad_out[b, i, j, oc]
                    
                    # Iterate over the kernel dimensions and channels
                    for ic in range(output_channels):
                        # Iterate over the kernel height
                        for di in range(kernel_height):
                            # Iterate over the kernel width
                            for dj in range(kernel_width):
                                # Compute the gradient of the input
                                grad_x[b, i0+di, j0+dj, ic] += w[oc, ic, di, dj] * go