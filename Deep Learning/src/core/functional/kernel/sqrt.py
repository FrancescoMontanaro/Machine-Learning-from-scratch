import math
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def sqrt_gradient(og_flat, x_data_flat, x_grad_flat) -> None:
    """
    Computes the gradient of the square root operation with respect to the input tensor.
    
    Parameters:
    - og_flat (np.ndarray): Gradient of the output scalar.
    - x_data_flat (np.ndarray): Flattened input tensor.
    - x_grad_flat (np.ndarray): Gradient of the flattened input tensor.
    """
    
    # Iterate over the flattened tensor
    for i in prange(x_grad_flat.size):
        # Compute the gradient of the square root operation
        x_grad_flat[i] += og_flat[i] / (2.0 * math.sqrt(x_data_flat[i]))