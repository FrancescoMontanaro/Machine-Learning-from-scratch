import numpy as np
from numba import njit, prange

from .reductions import reduce_to_shape


@njit(parallel=True, fastmath=True)
def elementwise_sub(a_view: np.ndarray, b_view: np.ndarray, out: np.ndarray) -> None:
    """
    Elementwise subtraction of two arrays.
    
    Parameters:
    - a_view (np.ndarray): First input array
    - b_view (np.ndarray): Second input array
    - out (np.ndarray): Output array to store the result
    """
    
    # Iterate over the flattened arrays
    for i in prange(out.size):
        # Perform elementwise subtraction
        out.flat[i] = a_view.flat[i] - b_view.flat[i]


@njit(fastmath=True)
def sub_forward(a_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
    """
    Forward pass for elementwise subtraction.
    
    Parameters:
    - a_data (np.ndarray): First input array
    - b_data (np.ndarray): Second input array
    - out (np.ndarray): Output array to store the result
    """
    
    # Extract the shapes of the input arrays
    out_shape = np.broadcast_shapes(a_data.shape, b_data.shape)
    
    # Create an output array with the broadcasted shape
    out = np.empty(out_shape, dtype=a_data.dtype)
    
    # Perform elementwise subtraction
    elementwise_sub(np.broadcast_to(a_data, out_shape), np.broadcast_to(b_data, out_shape), out)
    
    # Return the output array
    return out


@njit(fastmath=True)
def sub_gradient(out_grad: np.ndarray, a_shape: tuple, b_shape: tuple) -> tuple:
    """
    Computes the gradients for the inputs of the elementwise subtraction operation.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output
    - a_shape (tuple): Shape of the first input array
    - b_shape (tuple): Shape of the second input array
    - out (np.ndarray): Output array to store the result
    
    Returns:
    - tuple: Gradients for the first and second input arrays
    """
    
    # Compute the gradients for the first and second input arrays
    grad_a = reduce_to_shape(out_grad, a_shape)
    grad_b = reduce_to_shape(-out_grad, b_shape)
    
    # Return the gradients
    return grad_a, grad_b