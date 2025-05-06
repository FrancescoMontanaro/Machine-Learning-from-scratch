import math
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def sqrt_forward(x_data: np.ndarray) -> np.ndarray:
    """
    Computes the forward pass of the square root operation.
    
    Parameters:
    - x_data (np.ndarray): Input tensor.
    
    Returns:
    - np.ndarray: Output tensor containing the square root of each element in the input tensor.
    """
    
    # Create a flattened view of the input tensor
    x_data_flat = x_data.flatten()
    
    # Create an empty array for the output tensor with the same shape as the input tensor
    out_flat = np.empty_like(x_data_flat)

    # Iterate over the flattened tensor
    for i in prange(x_data_flat.size):
        # Compute the square root of each element
        out_flat[i] = np.sqrt(x_data_flat[i])

    # Reshape the output tensor to match the input tensor's shape
    return out_flat.reshape(x_data.shape)


@njit(parallel=True, fastmath=True)
def sqrt_backward(out_grad: np.ndarray, out_buffer: np.ndarray, x_data: np.ndarray) -> None:
    """
    Computes the backward pass of the square root operation with respect to the input tensor.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output tensor.
    - out_buffer (np.ndarray): Gradient of the input tensor.
    - x_data (np.ndarray): Input tensor.
    """
    
    # Create a flattened view of the input tensor
    out_grad_flat = out_grad.ravel()
    x_data_flat = x_data.flatten()
    out_buffer_flat = out_buffer.flatten()
    
    # Iterate over the flattened tensor
    for i in prange(out_buffer_flat.size):
        # Compute the gradient of the square root operation
        out_buffer_flat[i] += out_grad_flat[i] / (2.0 * np.sqrt(x_data_flat[i]))