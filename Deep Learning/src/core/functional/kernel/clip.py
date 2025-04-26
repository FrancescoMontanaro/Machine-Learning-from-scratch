import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def clip_forward(x_flat: np.ndarray, minv: float, maxv: float, out_flat: np.ndarray) -> None:
    """
    Clips the values in a flattened tensor to a specified range.
    
    Parameters:
    - x_flat (np.ndarray): Flattened input tensor.
    - minv (float): Minimum value for clipping.
    - maxv (float): Maximum value for clipping.
    - out_flat (np.ndarray): Flattened output tensor.
    """
    
    # Iterate over the flattened tensor
    for i in prange(x_flat.size):
        # Extract the value
        v = x_flat[i]
        
        # Clip the value to the specified range
        out_flat[i] = v if (v >= minv and v <= maxv) else (minv if v < minv else maxv)


@njit(parallel=True, fastmath=True)
def clip_gradient(x_flat: np.ndarray, out_grad_flat: np.ndarray, x_grad_flat: np.ndarray, minv: float, maxv: float) -> None:
    """
    Computes the gradient of the clipping operation with respect to the input tensor.
    
    Parameters:
    - x_flat (np.ndarray): Flattened input tensor.
    - out_grad_flat (np.ndarray): Gradient of the output tensor.
    - x_grad_flat (np.ndarray): Gradient of the flattened input tensor.
    - minv (float): Minimum value for clipping.
    - maxv (float): Maximum value for clipping.
    """
    
    # Iterate over the flattened tensor
    for i in prange(x_flat.size):
        # Extract the value
        v = x_flat[i]
        
        # If the value is within the clipping range, propagate the gradient
        if v >= minv and v <= maxv:
            # Propagate the gradient
            x_grad_flat[i] += out_grad_flat[i]