import numpy as np
from typing import List, Union


def split_forward(tensor_data: np.ndarray, indices_or_sections: Union[int, List[int]], axis: int) -> List[np.ndarray]:
    """
    Forward pass for splitting a tensor into multiple sub-tensors along a specified axis.

    Parameters:
    - tensor_data (np.ndarray): Input tensor to be split.
    - indices_or_sections (Union[int, List[int]]): If an integer, it indicates the number of equal splits to make.
    - axis (int): Axis along which to split the tensor.
    """

    # Check if the input tensor is empty
    if tensor_data.size == 0:
        raise ValueError("The input tensor must contain at least one element.")

    # Check if the indices_or_sections are valid
    if not indices_or_sections:
        raise ValueError("The indices_or_sections list must contain at least one index.")

    # Split the tensor along the specified axis
    return np.split(tensor_data, indices_or_sections, axis=axis)


def split_backward(out_grads: List[np.ndarray], out_buffer: np.ndarray, axis: int) -> None:
    """
    Backward pass for splitting a tensor into multiple sub-tensors along a specified axis.
    The gradient of the original tensor is obtained by concatenating all output gradients
    along the same axis used for splitting.

    Parameters:
    - out_grads (List[np.ndarray]): List of gradients for each sub-tensor output.
    - out_buffer (np.ndarray): Buffer to store the gradient with respect to the original input tensor.
    - axis (int): Axis along which the tensor was split.

    Returns:
    - np.ndarray: Gradient with respect to the original input tensor.
    """
    
    # Check if the output gradients list is valid
    if not out_grads:
        raise ValueError("The list of output gradients must not be empty.")
    
    # The backward of split is concatenation
    out_buffer[:] = np.concatenate(out_grads, axis=axis)