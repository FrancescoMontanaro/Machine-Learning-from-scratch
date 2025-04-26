from numba import njit, prange


@njit(parallel=True, fastmath=True)
def concat_forward(parts_flat, offsets, out_flat) -> None:
    """
    Concatenate multiple tensor segments into a single tensor.
    
    Parameters:
    - parts_flat (np.ndarray): Flattened segments of tensors to concatenate.
    - offsets (np.ndarray): Offsets for each segment in the concatenated tensor.
    - out_flat (np.ndarray): Output tensor to store the concatenated result.
    """
    
    # Iterate over the offsets to determine the start and end of each segment
    for p in prange(len(offsets)-1):
        # Get the start and end indices for the current segment
        start = offsets[p]
        end = offsets[p+1]
        
        # Iterate over the range of the current segment
        for i in range(start, end):
            # Copy the segment from parts_flat to out_flat
            out_flat[i] = parts_flat[i]


@njit(parallel=True, fastmath=True)
def concat_gradient(offsets, out_grad_flat, x_grad_flat, part_index) -> None:
    """
    Computes the gradient of the concatenation operation.
    
    Parameters:
    - offsets (np.ndarray): Offsets for each segment in the concatenated tensor.
    - out_grad_flat (np.ndarray): Gradient of the output tensor, flattened.
    - x_grad_flat (np.ndarray): Gradient of the input tensor, flattened.
    - part_index (int): Index of the segment for which to compute the gradient.
    """
    
    # Get the start and end indices for the current segment
    start = offsets[part_index]
    end = offsets[part_index+1]
    
    # Iterate over the range of the current segment
    for i in prange(start, end):
        # Add the gradient of the output tensor to the gradient of the input tensor
        x_grad_flat[i - start] += out_grad_flat[i]