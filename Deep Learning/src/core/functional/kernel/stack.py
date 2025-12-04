import numpy as np
from typing import List


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


def stack_backward(out_grad: np.ndarray, out_buffer: np.ndarray, axis: int, idx: int) -> None:
    """
    Backward pass for stacking tensors along a specified axis.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output tensor.
    - out_buffer (np.ndarray): Buffer to store the gradient for the specific tensor.
    - axis (int): Axis along which the tensors were stacked.
    - idx (int): Index of the tensor in the stack to which the gradient should be applied.
    """
    
    # Normalize the axis if negative
    if axis < 0:
        axis += out_grad.ndim
    if axis < 0 or axis >= out_grad.ndim:
        raise ValueError("Axis out of bounds.")

    # Check if the output gradient and buffer have compatible dimensions
    if out_grad.ndim != out_buffer.ndim + 1:
        # Raise an error if dimensions do not match
        raise ValueError("Mismatch between out_grad and out_buffer dimensions.")

    # Use np.take to select the gradient for the specific tensor along the stack axis
    src = np.take(out_grad, idx, axis=axis)

    # Check if the shape of the source matches the output buffer
    if src.shape != out_buffer.shape:
        # Raise an error if shapes do not match
        raise ValueError(f"Shape mismatch after slicing. src.shape={src.shape}, out_buffer.shape={out_buffer.shape}")

    # Copy the sliced gradient to the output buffer
    np.copyto(out_buffer, src)