import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def exp_gradient(out_data_flat: np.ndarray, out_grad_flat: np.ndarray, x_grad_flat: np.ndarray) -> None:
    """
    Computes the gradient of the exponential operation with respect to the input tensor.
    
    Parameters:
    - out_data_flat (np.ndarray): Flattened output tensor.
    - out_grad_flat (np.ndarray): Gradient of the output tensor.
    - x_grad_flat (np.ndarray): Gradient of the flattened input tensor.
    """
    
    # Iterate over the flattened tensor
    for i in prange(x_grad_flat.size):
        # Compute the gradient of the exponential operation
        x_grad_flat[i] += out_data_flat[i] * out_grad_flat[i]