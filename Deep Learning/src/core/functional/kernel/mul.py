import numpy as np
from numba import vectorize

from .reductions import reduce_to_shape


@vectorize(["float32(float32, float32)"], fastmath=True)
def mul_forward(a_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
    """
    Forward pass for elementwise multiplication.
    
    Parameters:
    - a_data (np.ndarray): First input array
    - b_data (np.ndarray): Second input array
    
    Returns:
    - np.ndarray: Result of the elementwise multiplication
    """
    
    # Perform elementwise multiplication of two arrays
    return a_data * b_data


def mul_backward_a(out_grad: np.ndarray, out_buffer: np.ndarray, target_shape: tuple, b_data: np.ndarray) -> None:
    """
    Computes the gradients for the inputs of the elementwise multiplication operation.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output
    - out_buffer (np.ndarray): Buffer to store the result
    - target_shape (tuple): Shape of the target output
    - b_data (np.ndarray): Second input array
    """
    
    # Compute the gradient for the first input array
    out_buffer += reduce_to_shape(out_grad * b_data, target_shape)


def mul_backward_b(out_grad: np.ndarray, out_buffer: np.ndarray, target_shape: tuple, a_data: np.ndarray) -> None:
    """
    Computes the gradients for the inputs of the elementwise multiplication operation.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output
    - out_buffer (np.ndarray): Buffer to store the result
    - target_shape (tuple): Shape of the target output
    - a_data (np.ndarray): First input array
    """
    
    # Compute the gradients for the first and second input arrays
    out_buffer += reduce_to_shape(out_grad * a_data, target_shape)