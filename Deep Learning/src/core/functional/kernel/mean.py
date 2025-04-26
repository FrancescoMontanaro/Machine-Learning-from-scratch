from numba import njit, prange


@njit(parallel=True, fastmath=True)
def mean_flat_backward(out_grad_scalar, x_grad_flat, inv_count) -> None:
    """
    Computes the gradient of the mean operation with respect to the input tensor.
    
    Parameters:
    - out_grad_scalar (np.ndarray): Gradient of the output scalar.
    - x_grad_flat (np.ndarray): Gradient of the flattened input tensor.
    - inv_count (float): Inverse count for normalization.
    """
    
    # Extract the gradient of the output scalar
    val = out_grad_scalar[0] * inv_count
    
    # Iterate over the flattened tensor and propagate the gradient
    for i in prange(x_grad_flat.size):
        # Add the gradient to each element of the input tensor
        x_grad_flat[i] += val