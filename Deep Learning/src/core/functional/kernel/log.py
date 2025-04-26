from numba import njit, prange


@njit(parallel=True)
def log_gradient(x_data_flat, out_grad_flat, x_grad_flat) -> None:
    """
    Computes the gradient of the logarithm operation with respect to the input tensor.
    
    Parameters:
    - x_data_flat (np.ndarray): Flattened input tensor.
    - out_grad_flat (np.ndarray): Gradient of the output tensor.
    - x_grad_flat (np.ndarray): Gradient of the flattened input tensor.
    """
    
    # Iterate over the flattened tensor
    for i in prange(x_grad_flat.size):
        # Compute the gradient of the logarithm operation
        x_grad_flat[i] += out_grad_flat[i] / x_data_flat[i]