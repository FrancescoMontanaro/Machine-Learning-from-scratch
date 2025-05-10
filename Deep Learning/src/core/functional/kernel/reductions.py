import numpy as np
from numba import guvectorize, float32


@guvectorize([(float32[:], float32[:])], '(n)->()', fastmath=True)
def sum_1d(in_arr: np.ndarray, out_scalar: np.ndarray) -> None:
    """
    Computes the sum of a 1D array and stores the result in a scalar.
    
    Parameters:
    - in_arr (np.ndarray): Input array of shape (n,)
    - out_scalar (np.ndarray): Output scalar to store the sum
    """
    
    # Initialize the output scalar
    s = 0.0
    
    # Iterate over the elements of the input array
    for i in range(in_arr.shape[-1]):
        # Accumulate the sum
        s += in_arr[(..., i)]
        
    # Store the result in the output scalar
    out_scalar[()] = s


def reduce_to_shape(x: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Function to reduce an array to a specified shape by summing over dimensions that are broadcasted.
    
    Parameters:
    - x (np.ndarray): Input array to be reduced
    - target_shape (tuple): Target shape to reduce to
    """
    
    # Broadcast the input array to the target shape
    full_shape = np.broadcast_shapes(x.shape, target_shape)
    arr = np.broadcast_to(x, full_shape)

    # Pad target_shape left with ones to match ndim
    pad = len(full_shape) - len(target_shape)
    full_target = (1,) * pad + tuple(target_shape)

    # Identify axes to reduce: where full_target[i]==1 and full_shape[i]!=1
    axes = [i for i in range(len(full_shape)) if full_target[i] == 1 and full_shape[i] != 1]

    # Reduce each axis in descending order
    for ax in sorted(axes, reverse=True):
        arr = np.moveaxis(arr, ax, -1)
        d, rest = arr.shape[-1], arr.shape[:-1]
        flat = arr.reshape(-1, d)
        
        # Call the sum_1d function to compute the sum over the last axis
        summed = np.empty((flat.shape[0],), dtype=flat.dtype)
        sum_1d(flat, summed)
        
        # Reshape the summed array to the original shape
        arr = summed.reshape(rest)

    # Reshape the array to the target shape
    return arr.reshape(target_shape)