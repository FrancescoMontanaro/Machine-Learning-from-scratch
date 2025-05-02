import numpy as np
from numba import njit


@njit(fastmath=True)
def reduce_to_shape(x: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Reduces an array to a specified shape by summing over dimensions that are broadcasted.
    
    Parameters:
    - x (np.ndarray): Input array to be reduced
    - target_shape (tuple): Target shape to reduce to
    
    Returns:
    - np.ndarray: Reduced array with the specified shape
    """
    
    # Ensure is of the correct type
    target_dims = np.asarray(target_shape, dtype=np.int64)
    
    # Comute the difference in dimensions
    ndim_t = target_dims.size
    ndim_x = x.ndim
    diff = ndim_x - ndim_t

    # Iterate over the dimensions of x
    for ax in range(ndim_x):
        # If the dimension is not in the target shape, sum over it
        if ax - diff >= 0 and ndim_t > 0:
            # Extract the target dimension
            tgt_dim = target_dims[ax - diff]
        # If the dimension is in the target shape, set it to 1
        else:
            # If the dimension is not in the target shape, set it to 1
            tgt_dim = 1

        # If the dimension is not in the target shape, sum over it
        if tgt_dim == 1 and x.shape[ax] != 1:            
            # Expand the dimensions of the array
            x = np.expand_dims(np.asarray(np.sum(x, axis=ax)), axis=ax)

    # Reshape the array to the target shape
    return np.ascontiguousarray(x).reshape(target_shape)