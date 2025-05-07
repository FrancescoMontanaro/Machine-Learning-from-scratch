import numpy as np
from numba import vectorize

from .reductions import reduce_to_shape


@vectorize(["float32(float32, float32)"], fastmath=True)
def div_forward(a_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
    """
    Forward pass for elementwise division.
    
    Parameters:
    - a_data (np.ndarray): First input array
    - b_data (np.ndarray): Second input array

    Returns:
    - np.ndarray: Result of the elementwise division
    """
    
    # Perform elementwise division of two arrays
    return a_data / b_data


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