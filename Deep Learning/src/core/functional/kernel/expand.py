import numpy as np
from numba import njit, prange

from .reductions import reduce_to_shape


def expand_forward(x: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Expands (broadcasts) a tensor to a target shape.
    
    A value of -1 in the target shape means keeping the original size for that dimension.
    The tensor can only be expanded along dimensions of size 1, or new dimensions can be added.
    
    Parameters:
    - x (np.ndarray): Input array to expand
    - target_shape (tuple): Target shape to expand to. Use -1 to keep original size.
    
    Returns:
    - np.ndarray: Broadcasted array with the target shape
    
    Raises:
    - ValueError: If the shapes are not compatible for broadcasting
    """
    
    # Resolve -1 values in target_shape with actual dimensions from input
    x_shape = x.shape
    x_ndim = len(x_shape)
    target_ndim = len(target_shape)
    
    # The target shape must have at least as many dimensions as the input
    if target_ndim < x_ndim:
        raise ValueError(
            f"The number of dimensions in target_shape ({target_ndim}) must be >= "
            f"the number of dimensions in input ({x_ndim})"
        )
    
    # Pad the input shape with ones on the left to match target dimensions
    padded_x_shape = (1,) * (target_ndim - x_ndim) + x_shape
    
    # Resolve -1 values and validate the expansion
    resolved_shape = []
    for i, (target_dim, x_dim) in enumerate(zip(target_shape, padded_x_shape)):
        if target_dim == -1:
            # Keep the original dimension
            resolved_shape.append(x_dim)
        elif x_dim == 1:
            # Can expand from size 1 to any size
            resolved_shape.append(target_dim)
        elif x_dim == target_dim:
            # Dimensions match, no expansion needed
            resolved_shape.append(target_dim)
        else:
            raise ValueError(
                f"Cannot expand dimension {i} from size {x_dim} to size {target_dim}. "
                f"Expansion is only allowed from size 1 or when dimensions match."
            )
    
    resolved_shape = tuple(resolved_shape)
    
    # Reshape input to have the same number of dimensions as target
    x_reshaped = x.reshape(padded_x_shape)
    
    # Use numpy broadcast_to for the actual expansion
    return np.broadcast_to(x_reshaped, resolved_shape)


def expand_backward(out_grad: np.ndarray, x_grad: np.ndarray, original_shape: tuple) -> None:
    """
    Computes the gradient for the expand operation.
    
    The backward pass sums the gradients over dimensions that were expanded (broadcasted).
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output (expanded tensor)
    - x_grad (np.ndarray): Buffer to accumulate the gradient of the input tensor
    - original_shape (tuple): Original shape of the input tensor before expansion
    """
    
    # Reduce the gradient back to the original shape by summing over expanded dimensions
    grad_reduced = reduce_to_shape(out_grad, original_shape)
    
    # Accumulate the gradient
    x_grad += grad_reduced


@njit(parallel=True, fastmath=True)
def expand_backward_sum_axis(out_grad_flat: np.ndarray, x_grad_flat: np.ndarray, expand_factor: int) -> None:
    """
    Optimized backward pass for when expansion is along a single contiguous axis.
    Sums groups of `expand_factor` elements in the flattened gradient.
    
    Parameters:
    - out_grad_flat (np.ndarray): Flattened gradient of the output
    - x_grad_flat (np.ndarray): Flattened gradient buffer for the input
    - expand_factor (int): The factor by which the dimension was expanded
    """
    
    n = x_grad_flat.size
    
    for i in prange(n):
        s = 0.0
        base = i * expand_factor
        
        for j in range(expand_factor):
            s += out_grad_flat[base + j]
        
        x_grad_flat[i] += s
