import numpy as np
from numba import njit, prange


@njit(parallel=True)
def squeeze_gradient(out_grad_flat, x_grad_flat, shape, axes, new_shape) -> None:
    """
    Computes the gradient of the squeeze operation.
    
    Parameters:
    - out_grad_flat (np.ndarray): Gradient of the output tensor, flattened.
    - x_grad_flat (np.ndarray): Gradient of the input tensor, flattened.
    - shape (tuple): Original shape of the input tensor.
    - axes (list): Axes that were squeezed.
    - new_shape (tuple): New shape after squeezing.
    """
    
    # Extract dimensions of the tensors
    ndim = len(shape)
    
    # Create a list of axes to be squeezed
    strides = np.empty(ndim, np.int64)
    
    # Initialize strides
    s = 1
    
    # Iterate over the dimensions in reverse order
    for d in range(ndim-1, -1, -1):
        # Set the stride for the current dimension
        strides[d] = s
        
        # Compute the stride for the next dimension
        s *= shape[d]
        
    # Iterate over the output gradient
    for j in prange(out_grad_flat.size):
        # Extract the index of the squeezed dimension
        rem = j
        
        # Initialize a new index for the new shape
        idx_new = [0] * len(new_shape)
        
        # Iterate over the dimensions in reverse order
        for d in range(len(new_shape)-1, -1, -1):
            # Compute the index for the new shape
            idx_new[d] = rem % new_shape[d]
            
            # Update the remainder for the next dimension
            rem //= new_shape[d]
            
        # Create a new index for the original shape
        idx = []
        ai = 0
        
        # Iterate over the dimensions of the original shape
        for dim in range(ndim):
            # If the dimension is squeezed, append 0
            if dim in axes:
                # Append 0 for squeezed dimensions
                idx.append(0)
                
            # Otherwise, append the index from the new shape
            else:
                # Append the index from the new shape
                idx.append(idx_new[ai])
                
                # Increment the index for the new shape
                ai += 1
                
        # Initialize the linear index
        lin = 0
        
        # Compute the linear index for the original shape
        for d, v in enumerate(idx): 
            lin += v * strides[d]
            
        # Update the gradient for the original shape
        x_grad_flat[lin] += out_grad_flat[j]