import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def repeat_forward(x_flat: np.ndarray, repeats: int, out_flat: np.ndarray) -> None:
    """
    Repeat the elements of a 1D array.
    
    Parameters:
    - x_flat (np.ndarray): Input 1D array
    - repeats (int): Number of times to repeat each element
    - out_flat (np.ndarray): Output array to store the repeated elements
    """
    
    # Compute the size of the input array
    n = x_flat.size
    
    # Iterate over the input array
    for i in prange(n):
        # Get the value at the current index
        v = x_flat[i]
        
        # Compute the base index for the repeated elements
        base = i * repeats
        
        # Repeat the value and store it in the output array
        for r in range(repeats):
            # Store the repeated value in the output array
            out_flat[base + r] = v


@njit(parallel=True, fastmath=True)
def repeat_gradient(out_grad_flat: np.ndarray, repeats: int, x_grad_flat: np.ndarray) -> None:
    """
    Compute the gradient of the repeat operation.
    
    Parameters:
    - out_grad_flat (np.ndarray): Gradient of the output array
    - repeats (int): Number of times each element was repeated
    - x_grad_flat (np.ndarray): Gradient of the input array
    """
    
    # Compute the size of the input array
    n = x_grad_flat.size
    
    # Iterate over the input array
    for i in prange(n):
        # Initialize the gradient at the current index
        s = 0.0
        base = i * repeats
        
        # Iterate over the repeated elements
        for r in range(repeats):
            # Sum the gradients of the repeated elements
            s += out_grad_flat[base + r]
            
        # Store the accumulated gradient in the input gradient
        x_grad_flat[i] += s