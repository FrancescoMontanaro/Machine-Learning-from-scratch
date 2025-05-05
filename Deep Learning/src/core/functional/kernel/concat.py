import numpy as np
from numba import njit, prange
        
        
@njit(parallel=True, fastmath=True)
def concat_1d_forward(ts_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate a list of 1D tensors into a single 1D tensor.
    
    Parameters:
    - ts_list (list of np.ndarray): List of 1D tensors to concatenate.
    
    Returns:
    - tuple[np.ndarray, np.ndarray]: A tuple containing the concatenated tensor and an array of offsets.
    """
    
    # Extract the length of each tensor in the list
    n = len(ts_list)
    
    # Create the offsets array to store the cumulative sizes
    offsets = np.empty(n + 1, dtype=np.int64)
    
    # Initialize the first offset to 0
    offsets[0] = 0
    
    # Iterate through the tensors to calculate the offsets
    for i in range(n):
        # Calculate the offset for the next tensor
        offsets[i+1] = offsets[i] + ts_list[i].size
        
    # Calculate the total size
    tot = offsets[-1]

    # Create an empty array to hold the concatenated result
    out = np.empty(tot, dtype=ts_list[0].dtype)
    
    # Iterate through the tensors and copy their data into the output array
    for p in prange(n):
        # Calculate the start and end indices for the current tensor
        s, e = offsets[p], offsets[p+1]
        
        # Copy the data from the current tensor into the output array
        out[s:e] = ts_list[p].ravel()
        
    # Return the concatenated array and the offsets
    return out, offsets  
        
        
@njit(parallel=True, fastmath=True)
def concat_2d_forward(ts_list: list[np.ndarray], axis: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate a list of 2D tensors along the specified axis.
    
    Parameters:
    - ts_list (list of np.ndarray): List of 2D tensors to concatenate.
    - axis (int): Axis along which to concatenate (0 for rows, 1 for columns).
    
    Returns:
    - tuple[np.ndarray, np.ndarray]: A tuple containing the concatenated tensor and an array of offsets.
    """
    
    # Extract the length of each tensor in the list
    n = len(ts_list)
    
    # Create the offsets array to store the cumulative sizes
    offsets = np.empty(n+1, dtype=np.int64)
    
    # Initialize the first offset to 0
    offsets[0] = 0
    
    # Iterate through the tensors to calculate the offsets
    for i in range(n):
        # Calculate the offset for the next tensor
        offsets[i+1] = offsets[i] + ts_list[i].size
        
    # Calculate the total size
    tot = offsets[-1]

    # Create an empty array to hold the concatenated result
    flat = np.empty(tot, dtype=ts_list[0].dtype)
    
    # Iterate through the tensors and copy their data into the output array
    for p in prange(n):
        # Calculate the start and end indices for the current tensor
        s, e = offsets[p], offsets[p+1]
        
        # Copy the data from the current tensor into the output array
        flat[s:e] = ts_list[p].ravel()

    # Reshape the flat array into the desired shape based on the axis
    if axis == 0:
        # Concatenate along rows
        rows = 0
        cols = ts_list[0].shape[1]
        
        # Iterate through the tensors to calculate the total number of rows
        for i in range(n):
            # Calculate the number of rows for the current tensor
            rows += ts_list[i].shape[0]
            
        # Reshape the flat array into the desired shape
        out = flat.reshape(rows, cols)
    else:
        # Concatenate along columns
        rows = ts_list[0].shape[0]
        cols = 0
        
        # Iterate through the tensors to calculate the total number of columns
        for i in range(n):
            # Calculate the number of columns for the current tensor
            cols += ts_list[i].shape[1]
            
        # Reshape the flat array into the desired shape
        out = flat.reshape(rows, cols)

    # Return the concatenated array and the offsets
    return out, offsets


@njit(parallel=True, fastmath=True)
def concat_backward(out_grad: np.ndarray, out_buffer: np.ndarray, offsets: np.ndarray, idx: int) -> None:
    """
    Backward pass for the concatenation operation.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output tensor.
    - out_buffer (np.ndarray): Buffer to store the gradients for each input tensor.
    - offsets (np.ndarray): Offsets for each input tensor.
    - idx (int): Index of the input tensor to which the gradient is being copied.
    """
    
    # Compute the start index for the current tensor
    start = offsets[idx]

    # viste piatte per la copia parallela
    grad_flat = out_grad.ravel()
    buf_flat = out_buffer.ravel()

    # Iterate through the gradient of the output tensor and copy it to the buffer
    for i in prange(buf_flat.size):
        # Copy the gradient from the output tensor to the buffer
        buf_flat[i] += grad_flat[start + i]