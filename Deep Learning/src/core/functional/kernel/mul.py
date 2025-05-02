import numpy as np
from numba import njit, prange

from .reductions import reduce_to_shape


@njit(parallel=True, fastmath=True)
def elementwise_mul(a_view: np.ndarray, b_view: np.ndarray, out: np.ndarray) -> None:
    """
    Elementwise multiplication of two arrays.
    
    Parameters:
    - a_view (np.ndarray): First input array
    - b_view (np.ndarray): Second input array
    - out (np.ndarray): Output array to store the result
    """
    
    # Iterate over the flattened arrays
    for i in prange(out.size):
        # Perform elementwise multiplication
        out.flat[i] = a_view.flat[i] * b_view.flat[i]


@njit(fastmath=True)
def mul_forward(a_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
    """
    Forward pass for elementwise multiplication.
    
    Parameters:
    - a_data (np.ndarray): First input array
    - b_data (np.ndarray): Second input array
    - out (np.ndarray): Output array to store the result
    """
    
    # Extract the shapes of the input arrays
    out_shape = np.broadcast_shapes(a_data.shape, b_data.shape)
    
    # Create an output array with the broadcasted shape
    out = np.empty(out_shape, dtype=a_data.dtype)
    
    # Perform elementwise multiplication
    elementwise_mul(np.broadcast_to(a_data, out_shape), np.broadcast_to(b_data, out_shape), out)
    
    # Return the output array
    return out


@njit(fastmath=True)
def mul_gradient(out_grad: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple:
    """
    Computes the gradients for the inputs of the elementwise multiplication operation.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output
    - a (np.ndarray): First input array
    - b (np.ndarray): Second input array
    - out (np.ndarray): Output array to store the result
    
    Returns:
    - tuple: Gradients for the first and second input arrays
    """
    
    # Compute the gradients for the first and second input arrays
    grad_a = reduce_to_shape(out_grad * b, a.shape)
    grad_b = reduce_to_shape(out_grad * a, b.shape)
    
    # Return the gradients
    return grad_a, grad_b