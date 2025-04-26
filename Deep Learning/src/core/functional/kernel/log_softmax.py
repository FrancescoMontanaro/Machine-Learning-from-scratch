import math
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def log_softmax_forward(data_flat, out_flat, M, K) -> None:
    """
    Computes the log softmax of the input data.
    
    Parameters:
    - data_flat: 1D array of input data
    - out_flat: 1D array to store the output
    - M: number of samples
    - K: number of classes/features per sample
    """
    
    # Iterate over each sample
    for i in prange(M):
        # Calculate the base index for the current sample
        base = i * K
        maxv = -1e30
        
        # Find the maximum value in the current sample
        for j in range(K):
            # Access the data using the base index and offset by j
            v = data_flat[base + j]
            
            # Update maxv if the current value is greater
            if v > maxv:
                maxv = v
                
        # Initialize the sum variable
        s = 0.0
        
        # Iterate over each class/feature
        for j in range(K):
            # Compute the exponential of the difference between the current value and maxv
            s += math.exp(data_flat[base + j] - maxv)
            
        # Compute the logarithm of the sum
        log_s = math.log(s)
        
        # Iterate again to compute the log softmax values
        for j in range(K):
            # Store the log softmax value in the output array
            out_flat[base + j] = data_flat[base + j] - maxv - log_s


@njit(parallel=True, fastmath=True)
def log_softmax_gradient(out_flat, grad_out_flat, grad_x_flat, M, K) -> None:
    """
    Computes the gradient of the log softmax function.
    
    Parameters:
    - out_flat: 1D array of log softmax output
    - grad_out_flat: 1D array of gradients from the next layer
    - grad_x_flat: 1D array to store the gradients with respect to the input
    - M: number of samples
    - K: number of classes/features per sample
    """
    
    # Iterate over each sample
    for i in prange(M):
        # Calculate the base index for the current sample
        base = i * K
        
        # Initialize the sum variable
        s = 0.0
        
        # Compute the sum of the gradients
        for j in range(K):
            # Access the gradient using the base index and offset by j
            s += grad_out_flat[base + j]
            
        # Iterate again to compute the gradient with respect to the input
        for j in range(K):
            # Update the gradient with respect to the input
            grad_x_flat[base + j] += grad_out_flat[base + j] - math.exp(out_flat[base + j]) * s