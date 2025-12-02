import numpy as np
from typing import Tuple


def top_k_forward(x: np.ndarray, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the k largest (or smallest) elements along a given dimension.
    
    Parameters:
    - x (np.ndarray): Input array
    - k (int): Number of top elements to return
    - dim (int): Dimension along which to find top k elements (default: -1, last dimension)
    - largest (bool): If True, return k largest elements; if False, return k smallest (default: True)
    - sorted (bool): If True, return elements in sorted order (default: True)
    
    Returns:
    - Tuple[np.ndarray, np.ndarray]: 
        - values: Top k values of shape (..., k, ...)
        - indices: Indices of top k values in original array, same shape as values
    """
    
    # Handle negative dimension
    ndim = x.ndim
    if dim < 0:
        dim = ndim + dim
    
    # Validate k
    if k > x.shape[dim]:
        raise ValueError(f"k ({k}) cannot be greater than dimension size ({x.shape[dim]})")
    
    # Move the target dimension to the last position for easier processing
    x_transposed = np.moveaxis(x, dim, -1)
    original_shape = x_transposed.shape
    
    # Reshape to 2D: (batch, dim_size)
    x_flat = x_transposed.reshape(-1, x_transposed.shape[-1])
    
    if largest:
        # Get indices of top k largest elements
        # Use argpartition for efficiency, then sort the top k
        if sorted:
            # Full argsort for sorted output
            indices_flat = np.argsort(-x_flat, axis=-1)[..., :k]
        else:
            # Use argpartition for unsorted (faster)
            indices_flat = np.argpartition(-x_flat, k - 1, axis=-1)[..., :k]
    else:
        # Get indices of top k smallest elements
        if sorted:
            indices_flat = np.argsort(x_flat, axis=-1)[..., :k]
        else:
            indices_flat = np.argpartition(x_flat, k - 1, axis=-1)[..., :k]
    
    # Gather values using advanced indexing
    batch_indices = np.arange(x_flat.shape[0])[:, None]
    values_flat = x_flat[batch_indices, indices_flat]
    
    # Reshape back to original shape with k instead of dim_size
    output_shape = list(original_shape[:-1]) + [k]
    values = values_flat.reshape(output_shape)
    indices = indices_flat.reshape(output_shape)
    
    # Move the dimension back to its original position
    values = np.moveaxis(values, -1, dim)
    indices = np.moveaxis(indices, -1, dim)

    # Return the top k values and their indices
    return values, indices


def top_k_backward(
    out_grad: np.ndarray, 
    out_buffer: np.ndarray, 
    indices: np.ndarray, 
    dim: int = -1
) -> None:
    """
    Computes the gradient for the top_k operation.
    
    The gradient is scattered back to the original positions using the saved indices.
    Only the top k elements receive gradients; all other positions get zero gradient.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output (top k values), shape (..., k, ...)
    - out_buffer (np.ndarray): Buffer to store the gradient of the input, shape (..., original_dim, ...)
    - indices (np.ndarray): Indices of the top k elements from forward pass, same shape as out_grad
    - dim (int): Dimension along which top_k was computed (default: -1)
    """
    
    # Handle negative dimension
    ndim = out_buffer.ndim
    if dim < 0:
        dim = ndim + dim
    
    # Move target dimension to the last position
    out_grad_transposed = np.moveaxis(out_grad, dim, -1)
    indices_transposed = np.moveaxis(indices, dim, -1)
    
    # Get shapes
    k = out_grad_transposed.shape[-1]
    original_dim_size = out_buffer.shape[dim]
    
    # Reshape to 2D for easier indexing
    batch_shape = out_grad_transposed.shape[:-1]
    batch_size = int(np.prod(batch_shape)) if batch_shape else 1
    
    out_grad_flat = out_grad_transposed.reshape(batch_size, k)
    indices_flat = indices_transposed.reshape(batch_size, k)
    
    # Create a temporary buffer in the transposed layout
    temp_buffer_shape = list(batch_shape) + [original_dim_size]
    temp_buffer_flat = np.zeros((batch_size, original_dim_size), dtype=out_grad.dtype)
    
    # Scatter gradients back to original positions
    batch_indices = np.arange(batch_size)[:, None]
    np.add.at(temp_buffer_flat, (batch_indices, indices_flat), out_grad_flat)
    
    # Reshape back and move axis to original position
    temp_buffer = temp_buffer_flat.reshape(temp_buffer_shape)
    temp_buffer = np.moveaxis(temp_buffer, -1, dim)
    
    # Accumulate into output buffer
    out_buffer += temp_buffer
