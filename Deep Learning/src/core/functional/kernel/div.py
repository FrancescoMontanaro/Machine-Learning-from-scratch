import numpy as np
from numba import njit, prange

from .reductions import reduce_to_shape


@njit(parallel=True, fastmath=True)
def elementwise_div(a_view: np.ndarray, b_view: np.ndarray, out: np.ndarray) -> None:
    """
    Elementwise division of two arrays.
    
    Parameters:
    - a_view (np.ndarray): First input array
    - b_view (np.ndarray): Second input array
    - out (np.ndarray): Output array to store the result
    """
    
    # Iterate over the flattened arrays
    for i in prange(out.size):
        # Perform elementwise division
        out.flat[i] = a_view.flat[i] / b_view.flat[i]


@njit(fastmath=True)
def div_forward(a_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
    """
    Forward pass for elementwise division.
    
    Parameters:
    - a_data (np.ndarray): First input array
    - b_data (np.ndarray): Second input array
    - out (np.ndarray): Output array to store the result
    """
    
    # Extract the shapes of the input arrays
    out_shape = np.broadcast_shapes(a_data.shape, b_data.shape)
    
    # Create an output array with the broadcasted shape
    out = np.empty(out_shape, dtype=a_data.dtype)
    
    # Perform elementwise division
    elementwise_div(np.broadcast_to(a_data, out_shape), np.broadcast_to(b_data, out_shape), out)
    
    # Return the output array
    return out


@njit(fastmath=True)
def div_backward_a(out_grad: np.ndarray, out_buffer: np.ndarray, target_shape: tuple, b_data: np.ndarray) -> None:
    """
    Computes the gradients for the inputs of the elementwise division operation.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output
    - out_buffer (np.ndarray): Buffer to store the result
    - target_shape (tuple): Shape of the target output
    - b_data (np.ndarray): Second input array
    """
    
    # Compute the gradient for the first input array
    out_buffer += reduce_to_shape(out_grad / b_data, target_shape)


@njit(fastmath=True)
def div_backward_b(out_grad: np.ndarray, out_buffer: np.ndarray, a_data: np.ndarray, b_data: np.ndarray) -> None:
    """
    Computes the gradients for the inputs of the elementwise division operation.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output
    - out_buffer (np.ndarray): Buffer to store the result
    - a_data (np.ndarray): First input array
    - b_data (np.ndarray): Second input array
    """
    
    # Compute the gradients for the first and second input arrays
    out_buffer += reduce_to_shape(-a_data * out_grad / (b_data ** 2), b_data.shape)