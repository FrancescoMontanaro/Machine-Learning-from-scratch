import numpy as np
from numba import vectorize

from .reductions import reduce_to_shape


@vectorize(["float32(float32, float32)"], fastmath=True)
def sub_forward(a_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
    """
    Forward pass for elementwise subtraction.
    
    Parameters:
    - a_data (np.ndarray): First input array
    - b_data (np.ndarray): Second input array
    
    Returns:
    - np.ndarray: Result of the elementwise subtraction
    """
    
    # Perform elementwise subtraction of two arrays
    return a_data - b_data


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