import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def relu_forward(x_flat: np.ndarray, out_flat: np.ndarray, n: int) -> None:
    """
    Computes the ReLU activation function.
    
    Parameters:
    - x_flat: 1D array of input data
    - out_flat: 1D array to store the output
    - n: number of elements in the input data
    """
    
    # Iterate over each element in the input data
    for i in prange(n):
        # Extract the value at index i
        v = x_flat[i]
        
        # Apply the ReLU function: max(0, x)
        out_flat[i] = v if v > 0.0 else 0.0


@njit(parallel=True, fastmath=True)
def relu_gradient(x_flat: np.ndarray, grad_out_flat: np.ndarray, grad_x_flat: np.ndarray, n: int) -> None:
    """
    Computes the gradient of the ReLU activation function.
    
    Parameters:
    - x_flat: 1D array of input data
    - grad_out_flat: 1D array of gradients from the next layer
    - grad_x_flat: 1D array to store the gradients with respect to the input
    - n: number of elements in the input data
    """
    
    # Iterate over each element in the input data
    for i in prange(n):
        # If the input value is greater than 0, propagate the gradient
        if x_flat[i] > 0.0:
            # Multiply the gradient from the next layer by 1
            grad_x_flat[i] += grad_out_flat[i]