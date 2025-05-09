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
    
    # Get the starting index for the gradient of the input tensor
    start = offsets[idx]
    
    # The size of the portion of the gradient to copy is the size of the input tensor's gradient buffer
    current_tensor_grad_size = out_buffer.size 

    # Flatten the output gradient and buffer for easier access
    grad_flat = out_grad.ravel()
    buf_flat = out_buffer.ravel()

    # Iterate over the range of the current tensor's gradient size
    for i in prange(current_tensor_grad_size):
        # Copy the gradient from the output gradient to the buffer
        buf_flat[i] = grad_flat[start + i]