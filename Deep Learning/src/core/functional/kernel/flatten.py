import numpy as np
from typing import Tuple


def flatten_forward(x: np.ndarray, start_dim: int = 0, end_dim: int = -1) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """
    Flattens a contiguous range of dimensions in the input array.
    
    Parameters:
    - x (np.ndarray): Input array
    - start_dim (int): First dimension to flatten (default: 0)
    - end_dim (int): Last dimension to flatten (default: -1, meaning last dimension)
    
    Returns:
    - Tuple[np.ndarray, Tuple[int, ...]]: Flattened array and original shape for backward pass
    """
    
    # Save the original shape for backward pass
    original_shape = x.shape
    
    # Handle negative indices
    ndim = x.ndim
    if start_dim < 0:
        start_dim = ndim + start_dim
    if end_dim < 0:
        end_dim = ndim + end_dim
    
    # Validate dimensions
    if start_dim < 0 or start_dim >= ndim:
        raise ValueError(f"start_dim {start_dim} is out of range for tensor with {ndim} dimensions")
    if end_dim < 0 or end_dim >= ndim:
        raise ValueError(f"end_dim {end_dim} is out of range for tensor with {ndim} dimensions")
    if start_dim > end_dim:
        raise ValueError(f"start_dim {start_dim} must be <= end_dim {end_dim}")
    
    # Compute the new shape
    # Keep dimensions before start_dim
    new_shape = list(original_shape[:start_dim])
    
    # Flatten dimensions from start_dim to end_dim (inclusive)
    flattened_size = int(np.prod(original_shape[start_dim:end_dim + 1]))
    new_shape.append(flattened_size)
    
    # Keep dimensions after end_dim
    new_shape.extend(original_shape[end_dim + 1:])
    
    # Reshape the array
    result = x.reshape(new_shape)
    
    return result, original_shape


def flatten_backward(out_grad: np.ndarray, out_buffer: np.ndarray, original_shape: Tuple[int, ...]) -> None:
    """
    Computes the gradient for the flatten operation by reshaping back to original shape.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output
    - out_buffer (np.ndarray): Buffer to store the gradient (must have original_shape)
    - original_shape (Tuple[int, ...]): Original shape of the input before flattening
    """
    
    # Reshape the gradient back to the original shape and accumulate
    out_buffer += out_grad.reshape(original_shape)
