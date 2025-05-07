import numpy as np
from numba import vectorize

from .reductions import reduce_to_shape


@vectorize(["float32(float32, float32)"], fastmath=True)
def add_forward(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes the elementwise addition of two arrays.
    
    Parameters:
    - a (np.ndarray): First input array
    - b (np.ndarray): Second input array
    
    Returns:
    - np.ndarray: Result of the elementwise addition
    """
    
    # Return the elementwise sum of the two arrays
    return a + b


def add_backward(out_grad: np.ndarray, out_buffer: np.ndarray, target_shape: tuple) -> None:
    """
    Computes the gradients for the inputs of the elementwise addition operation.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output
    - out_buffer (np.ndarray): Buffer to store the result
    - target_shape (tuple): Shape of the target output
    """
    
    # Compute the gradient for the first input array
    out_buffer += reduce_to_shape(out_grad, target_shape)