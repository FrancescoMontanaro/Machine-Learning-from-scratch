import numpy as np
from numba import njit, prange
from typing import Dict, Callable


# Cache for JIT-compiled functions
_concat_cache: Dict[int, Callable] = {}


def concat_forward_internal(rank: int) -> Callable:
    """
    Builds a JIT-compiled function for concatenating tensors of a specific rank.
    
    Parameters:
    - rank (int): The rank of the tensors to be concatenated.
    
    Returns:
    - Callable: A function that takes a list of tensors and an axis, and returns the concatenated tensor and offsets.
    """

    # Convert the rank to a constant for Numba
    R = rank

    @njit(parallel=True, fastmath=True)
    def concat_rd_forward(ts_list: list[np.ndarray], axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Concatenate a list of tensors of the same rank along the given axis.
        
        Parameters:
        - ts_list (list[np.ndarray]): List of tensors to concatenate.
        - axis (int): Axis along which to concatenate the tensors.
        
        Returns:
        - tuple[np.ndarray, np.ndarray]: A tuple containing the concatenated tensor and offsets.
        
        Raises:
        - ValueError: If the list of tensors is empty or if the axis is out of range.
        """
        
        # Extract the number of tensors and validate the axis
        n = len(ts_list)
        
        # Validate the number of tensors
        if n == 0:
            # Raise an error if the list is empty
            raise ValueError("ts_list must contain at least one tensor")
        
        # Validate the axis
        if axis < -R or axis >= R:
            # Raise an error if the axis is out of range
            raise ValueError("axis out of range")
        
        # If axis is negative, convert it to positive
        if axis < 0:
            # Convert negative axis to positive
            axis += R

        # Create an array to hold the offsets for each tensor and initialize the first offset
        offsets = np.empty(n + 1, dtype=np.int64)
        offsets[0] = 0
        
        # Initialize the offsets for each tensor
        for i in range(n):
            offsets[i + 1] = offsets[i] + ts_list[i].size
            
        # Extract the total size of the concatenated tensor
        tot = offsets[-1]

        # Create an empty array to hold the concatenated tensor
        flat = np.empty(tot, dtype=ts_list[0].dtype)

        # Iterate over the tensors and copy their data into the concatenated array
        for p in prange(n):
            s, e = offsets[p], offsets[p + 1]
            flat[s:e] = ts_list[p].ravel()

        # Build the shape of the concatenated tensor
        shape_elements_arr = np.empty(R, dtype=np.int64)

        # Iterate over the dimensions of the tensors
        for d_idx in range(R):
            # If the current dimension is the axis along which we are concatenating, 
            # sum the sizes of that dimension across all tensors
            if d_idx == axis:
                current_axis_dim_sum = 0
                for t_idx in range(n):
                    current_axis_dim_sum += ts_list[t_idx].shape[d_idx]
                shape_elements_arr[d_idx] = current_axis_dim_sum
                
            # If the current dimension is not the axis, take the size from the first tensor
            else:
                shape_elements_arr[d_idx] = ts_list[0].shape[d_idx]

        # Convert the shape elements to a tuple
        final_shape_tuple = None
        
        # Switch based on the rank to create the final shape tuple
        if R == 0:
            final_shape_tuple = ()
        elif R == 1:
            final_shape_tuple = (shape_elements_arr[0],)
        elif R == 2:
            final_shape_tuple = (shape_elements_arr[0], shape_elements_arr[1])
        elif R == 3:
            final_shape_tuple = (shape_elements_arr[0], shape_elements_arr[1], shape_elements_arr[2])
        elif R == 4:
            final_shape_tuple = (shape_elements_arr[0], shape_elements_arr[1], shape_elements_arr[2], shape_elements_arr[3])
        elif R == 5:
            final_shape_tuple = (shape_elements_arr[0], shape_elements_arr[1], shape_elements_arr[2], shape_elements_arr[3], shape_elements_arr[4])
        else:
            # If the rank is greater than 5, raise an error
            raise ValueError(f"Rank R={R} not supported by explicit tuple construction for reshape in JIT.")
        
        # Reshape the flat array to the final shape
        out = flat.reshape(final_shape_tuple)
        
        # Return the concatenated tensor and offsets
        return out, offsets

    # Return the JIT-compiled function
    return concat_rd_forward


def concat_forward(ts_list: list[np.ndarray], axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate a list of tensors of *any* rank along the given axis,
    returning (concatenated_tensor, offsets).

    The first call for a new rank triggers JIT compilation; subsequent
    calls hit the cached, machine-code version.
    """
    if not ts_list:
        raise ValueError("ts_list cannot be empty")

    rank = ts_list[0].ndim
    if rank not in _concat_cache:
        _concat_cache[rank] = concat_forward_internal(rank)

    return _concat_cache[rank](ts_list, axis)


@njit(parallel=True, fastmath=True, cache=True)
def concat_backward(out_grad: np.ndarray, out_buffer: np.ndarray, offsets: np.ndarray, idx: int) -> None:
    """
    Backward pass for the concatenation operation.
    
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output tensor.
    - out_buffer (np.ndarray): Buffer to store the gradient of the concatenated tensor.
    - offsets (np.ndarray): Offsets for each tensor in the concatenated tensor.
    - idx (int): Index of the tensor in the concatenated tensor.
    """
    
    # Extract the starting index and number of elements for the specified tensor
    start_idx_in_grad = offsets[idx]
    num_elements = out_buffer.size

    # Flatten out_grad to 1D for easier indexing
    out_grad_flat = out_grad.ravel()
    
    # Extract the segment of out_grad corresponding to the specified tensor
    source_segment = out_grad_flat[start_idx_in_grad : start_idx_in_grad + num_elements]

    # Check if out_buffer is C-contiguous
    if out_buffer.flags.c_contiguous:
        # Copy the data from source_segment to out_buffer
        buffer_flat_writable_view = out_buffer.ravel()
        for i in prange(num_elements):
            buffer_flat_writable_view[i] = source_segment[i]
    else:
        # Check if the size of out_buffer matches the number of elements
        if num_elements != out_buffer.size:
            raise ValueError("Size mismatch")

        # Copy the data from source_segment to out_buffer
        for i in range(num_elements):
            out_buffer.flat[i] = source_segment[i]