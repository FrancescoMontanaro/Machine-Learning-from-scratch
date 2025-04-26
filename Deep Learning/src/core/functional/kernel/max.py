from numba import njit

@njit(parallel=True)
def max_flat_forward(x_flat, out_scalar, idx_ptr) -> None:
    """
    Computes the maximum value in a flattened tensor and its index.
    
    Parameters:
    - x_flat (np.ndarray): Flattened input tensor.
    - out_scalar (np.ndarray): Output scalar to store the maximum value.
    - idx_ptr (np.ndarray): Pointer to store the index of the maximum value.
    """
    
    # Initialize the output scalar to the first element
    best = x_flat[0]
    best_i = 0
    
    # Iterate over the flattened tensor to find the maximum value and its index
    for i in range(1, x_flat.size):
        # Compare the current element with the best found so far
        v = x_flat[i]
        # if the current element is greater, update the best and index
        if v > best:
            # update the best value and index
            best = v; best_i = i
            
    # Store the maximum value and its index in the output parameters
    out_scalar[0] = best
    idx_ptr[0] = best_i


@njit(parallel=True)
def max_flat_gradient(idx_ptr, out_grad_scalar, x_grad_flat) -> None:
    """
    Computes the gradient of the max operation with respect to the input tensor.
    
    Parameters:
    - idx_ptr (np.ndarray): Pointer to the index of the maximum value.
    - out_grad_scalar (np.ndarray): Gradient of the output scalar.
    - x_grad_flat (np.ndarray): Gradient of the flattened input tensor.
    """
    
    # Extract the index of the maximum value
    i = idx_ptr[0]
    x_grad_flat[i] += out_grad_scalar[0]