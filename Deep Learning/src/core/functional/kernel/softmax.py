import math
from numba import njit, prange


@njit(parallel=True)
def softmax_forward(data_flat, out_flat, M, K) -> None:
    """
    Computes the softmax of the input data.
    
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
        
        # Initialize maxv to a very small value
        maxv = -1e30
        
        # Iterate over each class/feature
        for j in range(K):
            # Access the data using the base index and offset by j
            v = data_flat[base + j]
            
            # Update maxv if the current value is greater
            if v > maxv:
                maxv = v
                
        # Initialize the sum variable
        s = 0.0
        
        # Iterate over each class/feature again to compute the softmax values
        for j in range(K):
            # Compute the exponential of the difference between the current value and maxv
            ex = math.exp(data_flat[base + j] - maxv)
            
            # Store the softmax value in the output array
            out_flat[base + j] = ex
            s += ex
            
        # Normalize the softmax values by dividing by the sum
        inv_s = 1.0 / s
        for j in range(K):
            out_flat[base + j] *= inv_s


@njit(parallel=True)
def softmax_gradient(out_flat, grad_out_flat, grad_x_flat, M, K) -> None:
    """
    Computes the gradient of the softmax function.
    
    Parameters:
    - out_flat: 1D array of softmax output
    - grad_out_flat: 1D array of gradients from the next layer
    - grad_x_flat: 1D array to store the gradients with respect to the input
    - M: number of samples
    - K: number of classes/features per sample
    """
    
    # Iterate over each sample
    for i in prange(M):
        # Calculate the base index for the current sample
        base = i * K
        
        # Initialize the dot product variable
        dot = 0.0
        
        # Iterate over each class/feature to compute the dot product
        for j in range(K):
            # Compute the dot product of the softmax output and the gradient
            dot += out_flat[base + j] * grad_out_flat[base + j]
            
        # Iterate again to compute the gradients with respect to the input
        for j in range(K):
            # Update the gradient with respect to the input
            grad_x_flat[base + j] += out_flat[base + j] * (grad_out_flat[base + j] - dot)