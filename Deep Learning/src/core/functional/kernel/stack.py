import numpy as np
from typing import List
from numba import njit, prange


def stack_forward(tensors_data: List[np.ndarray], axis: int) -> np.ndarray:
    """
    Forward pass for stacking tensors along a specified axis.
    
    Parameters:
    - tensors_list_data (List[np.ndarray]): List of input tensors to be stacked.
    - axis (int): Axis along which to stack the tensors.
    """
    
    # Check if the input list is empty
    if not tensors_data:
        raise ValueError("The input list must contain at least one tensor.")

    # Stack the tensors along the specified axis
    return np.stack(tensors_data, axis=axis)


@njit(cache=True)
def _manual_unravel_index(flat_index: int, shape_tuple: tuple) -> tuple:
    # Extract the number of dimensions from the shape tuple
    ndims = len(shape_tuple)
    
    # Create a temporary array to hold the components of the index
    idx_comp = [0] * 6

    # Computhe the index components manually
    temp_flat_index = flat_index
    for i in range(ndims - 1, -1, -1):
        dim_val = shape_tuple[i]
        if dim_val == 0:
            # If the dimension is zero, we cannot proceed
            raise ValueError("La dimensione della forma non può essere zero in _manual_unravel_index.")
        idx_comp[i] = temp_flat_index % dim_val
        temp_flat_index //= dim_val

    # Return the index components as a tuple
    if ndims == 0:
        return ()
    elif ndims == 1:
        return (idx_comp[0],)
    elif ndims == 2:
        return (idx_comp[0], idx_comp[1])
    elif ndims == 3:
        return (idx_comp[0], idx_comp[1], idx_comp[2])
    elif ndims == 4:
        return (idx_comp[0], idx_comp[1], idx_comp[2], idx_comp[3])
    elif ndims == 5:
        return (idx_comp[0], idx_comp[1], idx_comp[2], idx_comp[3], idx_comp[4])
    elif ndims == 6:
        return (idx_comp[0], idx_comp[1], idx_comp[2], idx_comp[3], idx_comp[4], idx_comp[5])
    else:
        # Raise an error if the number of dimensions exceeds the supported limit
        raise ValueError(f"Maximum number of dimensions supported is 6, but got {ndims}.")


@njit(fastmath=True, cache=True)
def stack_backward(out_grad: np.ndarray, out_buffer: np.ndarray, axis: int, idx: int) -> None:
    """
    Backward pass for the stack operation.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output tensor.
    - out_buffer (np.ndarray): Buffer to store the gradient of the input tensors.
    - axis (int): Axis along which the tensors were stacked.
    - idx (int): Index of the tensor in the stack for which the gradient is being computed.
    """
    
    # If the axis is negative, adjust it to be positive
    if axis < 0:
        # Adjust the axis to be positive
        axis_adjusted = out_grad.ndim + axis
    else:
        # No adjustment needed
        axis_adjusted = axis

    # Check if the adjusted axis is within bounds
    if not (0 <= axis_adjusted < out_grad.ndim):
        # Raise an error if the axis is out of bounds
        raise ValueError("The axis is out of bounds for the given gradient tensor.")

    # Extract the shape and number of elements from the output buffer
    dest_shape = out_buffer.shape
    num_elements_dest = out_buffer.size

    # Check if the output buffer is a scalar
    if out_buffer.ndim == 0:
        # If the output buffer is a scalar, we need to handle it differently
        if out_grad.ndim == 1 and axis_adjusted == 0:
            # If out_grad is 1D and axis is 0, it can be directly indexed
            out_buffer[()] = out_grad[idx]
        else:
            # If the output buffer is a scalar and out_grad is not 1D, raise an error
            raise ValueError("Shapes not valid with out_buffer as scalar.")
        
        # If the output buffer is a scalar, we don't need to do anything else
        return

    # Check if the output gradient has the correct number of dimensions
    if out_grad.ndim != out_buffer.ndim + 1:
        # Raise an error if the dimensions do not match
        raise ValueError("Mismatch between out_grad and out_buffer dimensions.")

    # Iterate over the destination buffer using prange for parallel processing
    for p_idx in prange(num_elements_dest):
        # Unravel the flat index into multi-dimensional indices
        dest_multi_idx_tuple = _manual_unravel_index(p_idx, dest_shape)
        
        # Initialize a temporary list to hold the source indices
        current_dest_dim_idx = 0
        temp_src_indices_parts = [0] * out_grad.ndim

        # Iterate over the source dimensions to fill in the source indices
        for i_src_dim in range(out_grad.ndim):
            # Check if the current dimension is the axis we are stacking along
            if i_src_dim == axis_adjusted:
                # If it is, we set the index to the specified index
                temp_src_indices_parts[i_src_dim] = idx
            # Otherwise, we copy the index from the destination multi-index
            else:
                # Copy the index from the destination multi-index
                temp_src_indices_parts[i_src_dim] = dest_multi_idx_tuple[current_dest_dim_idx]
                current_dest_dim_idx += 1
        
        # Initialize a variable to hold the source multi-index
        src_multi_idx_as_tuple = None
        
        # Check the number of dimensions in the output gradient and create the source multi-index accordingly
        if out_grad.ndim == 1: 
            src_multi_idx_as_tuple = (temp_src_indices_parts[0],)
        elif out_grad.ndim == 2: 
            src_multi_idx_as_tuple = (temp_src_indices_parts[0], temp_src_indices_parts[1])
        elif out_grad.ndim == 3: 
            src_multi_idx_as_tuple = (temp_src_indices_parts[0], temp_src_indices_parts[1], temp_src_indices_parts[2])
        elif out_grad.ndim == 4: 
            src_multi_idx_as_tuple = (temp_src_indices_parts[0], temp_src_indices_parts[1], temp_src_indices_parts[2], temp_src_indices_parts[3])
        elif out_grad.ndim == 5: 
            src_multi_idx_as_tuple = (temp_src_indices_parts[0], temp_src_indices_parts[1], temp_src_indices_parts[2], temp_src_indices_parts[3], temp_src_indices_parts[4])
        elif out_grad.ndim == 6: 
            src_multi_idx_as_tuple = (temp_src_indices_parts[0], temp_src_indices_parts[1], temp_src_indices_parts[2], temp_src_indices_parts[3], temp_src_indices_parts[4], temp_src_indices_parts[5])
        elif out_grad.ndim == 7: 
            src_multi_idx_as_tuple = (temp_src_indices_parts[0], temp_src_indices_parts[1], temp_src_indices_parts[2], temp_src_indices_parts[3], temp_src_indices_parts[4], temp_src_indices_parts[5], temp_src_indices_parts[6])
        else:
            # Raise an error if the number of dimensions exceeds the supported limit
            raise ValueError(f"Maximum number of dimensions supported is 7, but got {out_grad.ndim}.")
         
        # Assign the gradient from the source multi-index to the destination multi-index
        out_buffer[dest_multi_idx_tuple] = out_grad[src_multi_idx_as_tuple]