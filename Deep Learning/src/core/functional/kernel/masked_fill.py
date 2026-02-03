import math
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def masked_fill_forward(x_flat: np.ndarray, mask_flat: np.ndarray, value: float, out_flat: np.ndarray) -> None:
    """
    Applies a masked fill operation on a flattened tensor.
    
    Parameters:
    - x_flat (np.ndarray): Flattened input tensor.
    - mask_flat (np.ndarray): Flattened boolean mask tensor.
    - value (float): Value to fill in the masked positions.
    - out_flat (np.ndarray): Flattened output tensor.
    """
    
    # Extract the size of the flattened tensors
    n = x_flat.size
    m = mask_flat.size
    
    # Iterate over the flattened tensor
    for i in prange(n):
        # If mask is True, fill with the specified value; otherwise, keep the original value
        out_flat[i] = value if mask_flat[i % m] else x_flat[i]
        
        
@njit(parallel=True, fastmath=True)
def masked_fill_forward_inf(x_flat, mask_flat, out_flat) -> None:
    """
    Applies a masked fill operation on a flattened tensor, filling with negative infinity.
    
    Parameters:
    - x_flat (np.ndarray): Flattened input tensor.
    - mask_flat (np.ndarray): Flattened boolean mask tensor.
    - out_flat (np.ndarray): Flattened output tensor.
    """   
    
    # Define the value to fill in the masked positions
    neg_inf = math.inf
    
    # Extract the size of the flattened tensors
    n = x_flat.size
    m = mask_flat.size
    
    # Iterate over the flattened tensor
    for i in prange(n):
        # If mask is True, fill with negative infinity; otherwise, keep the original value
        out_flat[i] = neg_inf if mask_flat[i % m] else x_flat[i]
        
        
@njit(parallel=True, fastmath=True)
def masked_fill_forward_neg_inf(x_flat, mask_flat, out_flat) -> None:
    """
    Applies a masked fill operation on a flattened tensor, filling with negative infinity.
    
    Parameters:
    - x_flat (np.ndarray): Flattened input tensor.
    - mask_flat (np.ndarray): Flattened boolean mask tensor.
    - out_flat (np.ndarray): Flattened output tensor.
    """   
    
    # Define the value to fill in the masked positions
    neg_inf = -math.inf
    
    # Extract the size of the flattened tensors
    n = x_flat.size
    m = mask_flat.size
    
    # Iterate over the flattened tensor
    for i in prange(n):
        # If mask is True, fill with negative infinity; otherwise, keep the original value
        out_flat[i] = neg_inf if mask_flat[i % m] else x_flat[i]


@njit(parallel=True, fastmath=True)
def masked_fill_gradient(mask_flat: np.ndarray, out_grad_flat: np.ndarray, x_grad_flat: np.ndarray) -> None:
    """
    Computes the gradient of the masked fill operation with respect to the input tensor.
    
    Parameters:
    - mask_flat (np.ndarray): Flattened boolean mask tensor.
    - out_grad_flat (np.ndarray): Gradient of the output tensor.
    - x_grad_flat (np.ndarray): Gradient of the flattened input tensor.
    """
    
    # Extract the size of the flattened tensors
    m = mask_flat.size
    n = out_grad_flat.size
    
    # Iterate over the flattened tensor
    for i in prange(n):
        # If mask is False, propagate the gradient (original value kept)
        # If mask is True, gradient is zero (value was replaced by constant)
        if not mask_flat[i % m]:
            # Propagate the gradient
            x_grad_flat[i] += out_grad_flat[i]