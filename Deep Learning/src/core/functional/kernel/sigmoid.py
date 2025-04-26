import math
from numba import njit, prange


@njit(parallel=True)
def sigmoid_forward(x_flat, out_flat, n) -> None:
    """
    Computes the sigmoid activation function.
    
    Parameters:
    - x_flat: 1D array of input data
    - out_flat: 1D array to store the output
    - n: number of elements in the input data
    """
    
    # Iterate over each element in the input data
    for i in prange(n):
        # Extract the value at index i
        v = x_flat[i]
        
        # Apply the sigmoid function: 1 / (1 + exp(-x))
        if v >= 0.0:
            # Avoid overflow for large negative values
            out_flat[i] = 1.0 / (1.0 + math.exp(-v))
            
        # If the value is negative, compute exp(x) and normalize
        else:
            # Avoid overflow for large positive values
            ev = math.exp(v)
            
            # Compute the sigmoid value
            out_flat[i] = ev / (1.0 + ev)


@njit(parallel=True)
def sigmoid_gradient(out_flat, grad_out_flat, grad_x_flat, n) -> None:
    """
    Computes the gradient of the sigmoid activation function.
    
    Parameters:
    - out_flat: 1D array of sigmoid output
    - grad_out_flat: 1D array of gradients from the next layer
    - grad_x_flat: 1D array to store the gradients with respect to the input
    - n: number of elements in the input data
    """
    
    # Iterate over each element in the input data
    for i in prange(n):
        # Compute the gradient of the sigmoid function
        grad_x_flat[i] += out_flat[i] * (1.0 - out_flat[i]) * grad_out_flat[i]