import numpy as np
from typing import Optional


def scatter_forward(
    x: np.ndarray, 
    dim: int, 
    index: np.ndarray, 
    src: np.ndarray,
    reduce: Optional[str] = None
) -> np.ndarray:
    """
    Scatter values from src into x at positions specified by index along the given dimension.
    
    This is equivalent to PyTorch's tensor.scatter_(dim, index, src).
    
    For a 3D tensor, the operation is:
        out[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
        out[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
        out[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
    
    Parameters:
    - x (np.ndarray): Input tensor to scatter into (will be copied)
    - dim (int): Dimension along which to scatter
    - index (np.ndarray): Indices where to scatter values. Must have same number of dimensions as x.
    - src (np.ndarray): Source values to scatter. Must have same shape as index.
    - reduce (Optional[str]): Reduction operation: None (replace), 'add', 'multiply'. Default: None
    
    Returns:
    - np.ndarray: Output tensor with scattered values
    """
    
    # Handle negative dimension
    ndim = x.ndim
    if dim < 0:
        dim = ndim + dim
    
    # Validate inputs
    if index.ndim != x.ndim:
        raise ValueError(f"index must have same number of dimensions as x ({x.ndim}), got {index.ndim}")
    if src.shape != index.shape:
        raise ValueError(f"src must have same shape as index ({index.shape}), got {src.shape}")
    
    # Create output copy
    out = x.copy()
    
    # Build index tuples for advanced indexing
    # We need to create indices for all dimensions
    idx_list = []
    for d in range(ndim):
        if d == dim:
            # Use the index array for the scatter dimension
            idx_list.append(index)
        else:
            # Create broadcasting indices for other dimensions
            shape = [1] * ndim
            shape[d] = index.shape[d]
            idx_list.append(np.arange(index.shape[d]).reshape(shape) * np.ones(index.shape, dtype=np.int64))
    
    # Convert to tuple for advanced indexing
    idx_tuple = tuple(idx_list)
    
    # Perform the scatter operation
    if reduce is None:
        out[idx_tuple] = src
    elif reduce == 'add':
        np.add.at(out, idx_tuple, src)
    elif reduce == 'multiply':
        np.multiply.at(out, idx_tuple, src)
    else:
        raise ValueError(f"Unknown reduce operation: {reduce}. Must be None, 'add', or 'multiply'")
    
    return out


def scatter_backward_x(
    out_grad: np.ndarray,
    out_buffer: np.ndarray,
    dim: int,
    index: np.ndarray,
    reduce: Optional[str] = None
) -> None:
    """
    Computes the gradient for the input tensor x in scatter operation.
    
    For scatter with replace (reduce=None): gradient flows everywhere except at scattered positions.
    For scatter with reduce='add': gradient flows everywhere (including scattered positions).
    For scatter with reduce='multiply': gradient depends on the values.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output
    - out_buffer (np.ndarray): Buffer to store gradient of x (same shape as x)
    - dim (int): Dimension along which scatter was performed
    - index (np.ndarray): Indices used in forward pass
    - reduce (Optional[str]): Reduction operation used in forward: None, 'add', 'multiply'
    """
    
    # Handle negative dimension
    ndim = out_buffer.ndim
    if dim < 0:
        dim = ndim + dim
    
    # Build index tuples for advanced indexing
    idx_list = []
    for d in range(ndim):
        if d == dim:
            idx_list.append(index)
        else:
            shape = [1] * ndim
            shape[d] = index.shape[d]
            idx_list.append(np.arange(index.shape[d]).reshape(shape) * np.ones(index.shape, dtype=np.int64))
    
    # Convert to tuple for advanced indexing
    idx_tuple = tuple(idx_list)
    
    # Perform the gradient scattering
    if reduce is None:
        # For replace operation: gradient flows to positions NOT overwritten
        # First, add all gradients
        out_buffer += out_grad
        # Then zero out the positions that were overwritten (they came from src, not x)
        out_buffer[idx_tuple] = 0
    elif reduce == 'add':
        # For add operation: gradient flows to all positions in x
        out_buffer += out_grad
    elif reduce == 'multiply':
        # For multiply: d(x * src)/dx = src, but this is complex to track
        # Generally requires saved src values - simplified version here
        out_buffer += out_grad
    else:
        raise ValueError(f"Unknown reduce operation: {reduce}")


def scatter_backward_src(
    out_grad: np.ndarray,
    out_buffer: np.ndarray,
    dim: int,
    index: np.ndarray,
    reduce: Optional[str] = None
) -> None:
    """
    Computes the gradient for the source tensor src in scatter operation.
    
    Gradient flows from the scattered positions back to src.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output
    - out_buffer (np.ndarray): Buffer to store gradient of src (same shape as index)
    - dim (int): Dimension along which scatter was performed
    - index (np.ndarray): Indices used in forward pass
    - reduce (Optional[str]): Reduction operation used in forward: None, 'add', 'multiply'
    """
    
    # Handle negative dimension
    ndim = out_grad.ndim
    if dim < 0:
        dim = ndim + dim
    
    # Build index tuples for advanced indexing
    idx_list = []
    for d in range(ndim):
        if d == dim:
            idx_list.append(index)
        else:
            shape = [1] * ndim
            shape[d] = index.shape[d]
            idx_list.append(np.arange(index.shape[d]).reshape(shape) * np.ones(index.shape, dtype=np.int64))
    
    # Convert to tuple for advanced indexing
    idx_tuple = tuple(idx_list)
    
    # Perform the gradient gathering
    if reduce is None or reduce == 'add':
        # Gradient at scattered positions flows back to src
        out_buffer += out_grad[idx_tuple]
    elif reduce == 'multiply':
        # For multiply: d(x * src)/dsrc = x (original x values needed)
        # Simplified version - just gather gradients
        out_buffer += out_grad[idx_tuple]
    else:
        # For unknown reduce operations, raise an error
        raise ValueError(f"Unknown reduce operation: {reduce}")