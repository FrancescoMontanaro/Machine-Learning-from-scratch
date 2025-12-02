import numpy as np


def gather_forward(x: np.ndarray, dim: int, index: np.ndarray) -> np.ndarray:
    """
    Gather values from x along a dimension specified by index.
    
    This is equivalent to PyTorch's torch.gather(x, dim, index).
    
    For a 3D tensor, the operation is:
        out[i][j][k] = x[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = x[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = x[i][j][index[i][j][k]]  # if dim == 2
    
    Parameters:
    - x (np.ndarray): Input tensor to gather from
    - dim (int): Dimension along which to gather
    - index (np.ndarray): Indices to gather. Must have same number of dimensions as x.
    
    Returns:
    - np.ndarray: Output tensor with gathered values, same shape as index
    """
    
    # Handle negative dimension
    ndim = x.ndim
    if dim < 0:
        dim = ndim + dim
    
    # Validate inputs
    if index.ndim != x.ndim:
        raise ValueError(f"index must have same number of dimensions as x ({x.ndim}), got {index.ndim}")
    
    # Build index tuples for advanced indexing
    idx_list = []
    for d in range(ndim):
        if d == dim:
            # Use the index array for the gather dimension
            idx_list.append(index)
        else:
            # Create broadcasting indices for other dimensions
            shape = [1] * ndim
            shape[d] = index.shape[d]
            idx_list.append(np.arange(index.shape[d]).reshape(shape) * np.ones(index.shape, dtype=np.int64))
    
    # Convert to tuple for advanced indexing
    idx_tuple = tuple(idx_list)
    
    # Perform the gather operation
    out = x[idx_tuple]
    
    return out


def gather_backward(
    out_grad: np.ndarray,
    out_buffer: np.ndarray,
    dim: int,
    index: np.ndarray
) -> None:
    """
    Computes the gradient for the input tensor x in gather operation.
    
    The gradient is scattered back from the output positions to the original input positions.
    This is essentially the inverse of gather - a scatter_add operation.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output, same shape as index
    - out_buffer (np.ndarray): Buffer to store gradient of x (same shape as original x)
    - dim (int): Dimension along which gather was performed
    - index (np.ndarray): Indices used in forward pass
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
    
    # Scatter add the gradients back to their original positions
    # Multiple gathered elements from the same position will have their gradients summed
    np.add.at(out_buffer, idx_tuple, out_grad)
