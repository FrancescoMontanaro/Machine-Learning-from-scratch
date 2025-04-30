import math
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def tanh_forward(x_flat: np.ndarray, out_flat: np.ndarray, n: int) -> None:
    """
    Computes the hyperbolic tangent activation function.
    
    Parameters:
    - x_flat: 1D array of input data
    - out_flat: 1D array to store the output
    - n: number of elements in the input data
    """
    
    # Iterate over each element in the input data
    for i in prange(n):
        # Apply the tanh function: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        out_flat[i] = math.tanh(x_flat[i])
        

@njit(parallel=True, fastmath=True)
def tanh_gradient(out_flat: np.ndarray, grad_out_flat: np.ndarray, grad_x_flat: np.ndarray, n: int) -> None:
    """
    Computes the gradient of the hyperbolic tangent activation function.
    
    Parameters:
    - out_flat: 1D array of tanh output
    - grad_out_flat: 1D array of gradients from the next layer
    - grad_x_flat: 1D array to store the gradients with respect to the input
    - n: number of elements in the input data
    """
    
    # Iterate over each element in the input data
    for i in prange(n):
        # Compute the gradient of the tanh function
        grad_x_flat[i] += (1.0 - out_flat[i] * out_flat[i]) * grad_out_flat[i]