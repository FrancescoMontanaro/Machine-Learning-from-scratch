import numpy as np
from numba import njit, prange

from .reductions import reduce_to_shape


@njit(parallel=True, fastmath=True)
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
    
    # Broadcast the input arrays to the output shape
    a_view = np.broadcast_to(a_data, out_shape)
    b_view = np.broadcast_to(b_data, out_shape)

    # Iterate over the flattened arrays
    for i in prange(out.size):
        # Perform elementwise subtraction
        out.flat[i] = a_view.flat[i] - b_view.flat[i]
    
    # Return the output array
    return out


@njit(fastmath=True)
def sub_backward_a(out_grad: np.ndarray, out_buffer: np.ndarray, target_shape: tuple) -> None:
    """
    Computes the gradients for the inputs of the elementwise subtraction operation.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output
    - out_buffer (np.ndarray): Buffer to store the result
    - target_shape (tuple): Shape of the target output
    """
    
    # Compute the gradients for the first and second input arrays
    out_buffer += reduce_to_shape(out_grad, target_shape)


@njit(fastmath=True)
def sub_backward_b(out_grad: np.ndarray, out_buffer: np.ndarray, target_shape: tuple) -> None:
    """
    Computes the gradients for the inputs of the elementwise subtraction operation.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output
    - out_buffer (np.ndarray): Buffer to store the result
    - target_shape (tuple): Shape of the target output
    """
    
    # Compute the gradients for the first and second input arrays
    out_buffer -= reduce_to_shape(out_grad, target_shape)