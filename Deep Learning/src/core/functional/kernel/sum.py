import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def sum_flat_forward(x_flat: np.ndarray, out_scalar: np.ndarray) -> None:
    """
    Computes the sum of all elements in a flattened tensor.
    
    Parameters:
    - x_flat (np.ndarray): Flattened input tensor.
    - out_scalar (np.ndarray): Output scalar to store the sum.
    """
    
    # Initialize the output scalar to zero
    tmp = 0.0
    
    # Iterate over the flattened tensor and compute the sum
    for i in prange(x_flat.size):
        # Add each element to the sum
        tmp += x_flat[i]
        
    # Store the result in the output scalar
    out_scalar[0] = tmp

@njit(parallel=True, fastmath=True)
def sum_flat_gradient(out_grad_scalar: np.ndarray, x_grad_flat: np.ndarray) -> None:
    """
    Computes the gradient of the sum operation with respect to the input tensor.
    
    Parameters:
    - out_grad_scalar (np.ndarray): Gradient of the output scalar.
    - x_grad_flat (np.ndarray): Gradient of the flattened input tensor.
    """
    
    # Extract the gradient of the output scalar
    val = out_grad_scalar[0]
    
    # Iterate over the flattened tensor and propagate the gradient
    for i in prange(x_grad_flat.size):
        # Add the gradient to each element of the input tensor
        x_grad_flat[i] += val