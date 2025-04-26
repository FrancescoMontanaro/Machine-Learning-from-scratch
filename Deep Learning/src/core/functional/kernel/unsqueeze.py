from numba import njit, prange

@njit(parallel=True)
def unsqueeze_gradient(out_grad_flat, x_grad_flat) -> None:
    """
    Computes the gradient of the unsqueeze operation.
    
    Parameters:
    - out_grad_flat (np.ndarray): Gradient of the output tensor, flattened.
    - x_grad_flat (np.ndarray): Gradient of the input tensor, flattened.
    """
    
    # Iterate over the output gradient
    for i in prange(x_grad_flat.size):
        # Add the gradient of the output tensor to the gradient of the input tensor
        x_grad_flat[i] += out_grad_flat[i]