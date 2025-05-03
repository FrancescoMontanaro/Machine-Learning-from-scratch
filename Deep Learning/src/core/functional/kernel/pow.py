import numpy as np
from numba import njit, prange
        
        
@njit(parallel=True, fastmath=True)
def pow_forward(t: np.ndarray, power: float) -> np.ndarray:
    """
    Forward pass for element
    
    Parameters:
    - t (np.ndarray): The input array to be raised to the power
    - power (float): Power to raise the elements of the first array
    
    Returns:
    - np.ndarray: Output array with the result of raising each element to the power
    """
    
    # Create an output array with the same shape as the input
    out = np.empty(t.shape, dtype=t.dtype)
    
    # Iterate over the flattened arrays
    for i in prange(out.size):
        # Perform elementwise raise to the power
        out.flat[i] = t.flat[i] ** power
    
    # Return the output array
    return out


@njit(fastmath=True)
def pow_gradient(out_grad: np.ndarray, t: np.ndarray, power: float) -> np.ndarray:
    """
    Computes the gradients for the inputs of the elementwise raise to the power operation.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output
    - t (np.ndarray): The input array to be raised to the power
    - power (float): Power to raise the elements of the first array
    
    Returns:
    - np.ndarray: Gradient of the input array
    """
    
    # Compute the gradient of the output with respect to the input
    return power * (t ** (power - 1)) * out_grad