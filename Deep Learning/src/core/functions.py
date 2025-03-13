import numpy as np
from typing import Optional, Tuple, List, Union, Type, TYPE_CHECKING, cast

from .registry import get_tensor_class
if TYPE_CHECKING: from .tensor import Tensor


def sum(x: 'Tensor', axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
    """
    Compute the sum of the tensor along the specified axis.
    
    Parameters:
    - x (Tensor): Input tensor
    - axis (int): Axis along which to compute the sum
    - keepdims (bool): Whether to keep the dimensions of the input tensor
    
    Returns:
    - Tensor: Sum of the tensor along the specified axis
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Compute the sum of the tensor along the specified axis
    out = Tensor(x.data.sum(axis=axis, keepdims=keepdims), requires_grad=x.requires_grad)

    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # If axis is None, broadcast the gradient to the shape of the input tensor
            if axis is None:
                # Broadcast the gradient to the shape of the input tensor
                out.grad = np.broadcast_to(out.grad, x.data.shape)
            else:
                # If axis is not None, expand the gradient along the specified axis if necessary
                if not keepdims:
                    # Expand the gradient along the specified axis
                    out.grad = np.expand_dims(out.grad, axis=axis)
                    
                # Broadcast the gradient to the shape of the input tensor
                out.grad = np.broadcast_to(out.grad, x.data.shape)
                
            # Update the gradient of the input tensor
            x.grad = x.grad + out.grad if x.grad is not None else out.grad
            
    # Store the backward function with respect to the sum operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x}
    
    # Return the output tensor
    return out


def max(x: 'Tensor', axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
    """
    Compute the maximum value of the tensor along the specified axis.
    
    Parameters:
    - x (Tensor): Input tensor
    - axis (int): Axis along which to compute the maximum value
    - keepdims (bool): Whether to keep the dimensions of the input tensor
    
    Returns:
    - Tensor: Maximum value of the tensor along the specified axis
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Compute the maximum value of the tensor along the specified axis
    out = Tensor(np.max(x.data, axis=axis, keepdims=keepdims), requires_grad=x.requires_grad)

    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Broadcast the gradient to the shape of the input tensor
            broadcasted_max = np.broadcast_to(out.data, x.data.shape)
            
            # Create a mask to identify the maximum elements
            mask = (x.data == broadcasted_max).astype(x.data.dtype)
            
            # Count the number of maximum elements
            count = np.sum(mask, axis=axis, keepdims=True) if axis is not None else np.sum(mask)
            
            # Distribute the gradient to the maximum elements
            grad_self = mask * (out.grad / count)
            
            # Update the gradient of the input tensor
            x.grad = x.grad + grad_self if x.grad is not None else grad_self

    # Store the backward function with respect to the maximum operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x}
    
    # Return the output tensor
    return out


def sqrt(x: 'Tensor') -> 'Tensor':
    """
    Compute the element-wise square root of the tensor.
    
    Parameters:
    - x (Tensor): Input tensor.
    
    Returns:
    - Tensor: Square root of the input tensor.
    
    Raises:
    - AssertionError: If the input is not a tensor.
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"

    # Compute the square root of the tensor
    out = Tensor(np.sqrt(x.data), requires_grad=x.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Avoid division by zero; assume x.data is non-negative.
            grad_self = out.grad / (2 * np.sqrt(x.data))
            
            # Update the gradient of the input tensor
            x.grad = x.grad + grad_self if x.grad is not None else grad_self

    # Store the backward function with respect to the square root operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x}
    
    # Return the output tensor
    return out


def mean(x: 'Tensor', axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
    """
    Compute the mean of the tensor along the specified axis.
    
    Parameters:
    - axis (Optional[Union[int, Tuple[int, ...]]]): Axis along which to compute the mean
    - keepdims (bool): Whether to keep the dimensions of the input tensor
    
    Returns:
    - Tensor: Mean of the tensor along the specified axis
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Compute the mean of the tensor along the specified axis
    out = Tensor(np.mean(x.data, axis=axis, keepdims=keepdims), requires_grad=x.requires_grad)

    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Determine the number of elements over which the mean is computed.
            if axis is None:
                # If axis is None, the mean is computed over all elements.
                count = x.data.size
            elif isinstance(axis, int):
                # If axis is an integer, the mean is computed over the elements in that axis.
                count = x.data.shape[axis]
            else:
                # If axis is a tuple, the mean is computed over the elements in the specified axes.
                count = 1
                for a in axis:
                    count *= x.data.shape[a]
                    
            # Scale the gradient by the reciprocal of the number of elements over which the mean is computed.
            grad_self = out.grad * (1.0 / count)
            
            # If axis is not None and keepdims is False, expand dimensions for proper broadcasting.
            if axis is not None and not keepdims:
                # If axis is an integer, expand the gradient along that axis.
                if isinstance(axis, int):
                    # Expand the gradient along the specified axis.
                    grad_self = np.expand_dims(grad_self, axis=axis)
                # If axis is a tuple, expand the gradient along each axis in the tuple.
                else:
                    # Sort the axes in ascending order to avoid conflicts.
                    for a in sorted(axis):
                        # Expand the gradient along the specified axis.
                        grad_self = np.expand_dims(grad_self, axis=a)
            
            # Broadcast the gradient to the original shape of self.data.
            grad_self = np.broadcast_to(grad_self, x.data.shape)
            
            # Update the gradient of the input tensor.
            x.grad = x.grad + grad_self if x.grad is not None else grad_self

    # Store the backward function with respect to the mean operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x}
    
    # Return the output tensor
    return out


def var(x: 'Tensor', axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
    """
    Compute the variance of the tensor along the specified axis (or axes).

    Parameters:
    - x (Tensor): Input tensor.
    - axis (Optional[Union[int, Tuple[int, ...]]]): Axis or axes along which to compute the variance.
     - keepdims (bool): Whether to keep the dimensions of the result.

    Returns:
    - Tensor: Variance of the tensor.
    
    Raises:
    - AssertionError: If the input is not a tensor.
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Compute the mean with keepdims=True to allow proper broadcasting.
    m = mean(x, axis=axis, keepdims=True)
    
    # Compute the squared difference between the tensor and the mean.
    diff = x - m
    
    # Compute the squared difference and take the mean along the specified axis.
    sq = diff * diff
    
    # Compute the variance by taking the mean of the squared difference.
    return mean(sq, axis=axis, keepdims=keepdims)


def exp(x: 'Tensor') -> 'Tensor':
    """
    Compute the exponential of the tensor.
    
    Parameters:
    - x (Tensor): Input tensor
    
    Returns:
    - Tensor: Exponential of the tensor
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Compute the exponential of the tensor
    out = Tensor(np.exp(x.data), requires_grad=x.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad:
            # Compute the gradient of the loss with respect to the current tensor
            x.grad = x.grad + out.data * out.grad if x.grad is not None else out.data * out.grad
            
    # Store the backward function with respect to the exponential operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x}
    
    # Return the output tensor
    return out


def log(x: 'Tensor') -> 'Tensor':
    """
    Computes the natural logarithm of the tensor.
    
    Returns:
    - Tensor: Natural logarithm of the tensor
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Forward pass: compute the natural logarithm
    out = Tensor(np.log(x.data), requires_grad=x.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Derivative of log(x) is 1/x; propagate the gradient accordingly.
            x.grad = x.grad + (out.grad / x.data) if x.grad is not None else (out.grad / x.data)
    
    # Store the backward function with respect to the natural logarithm operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x}
    
    # Return the output tensor
    return out


def transpose(x: 'Tensor', axes: Tuple[int]) -> 'Tensor':
    """
    Compute the transpose of the tensor.
    
    Parameters:
    - x (Tensor): Input tensor
    - axes (Tuple[int]): Permutation of the dimensions
    
    Returns:
    - Tensor: Transpose of the tensor
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Compute the transpose of the tensor
    out = Tensor(x.data.transpose(axes), requires_grad=x.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # Invert the axes
        inv_axes = np.argsort(axes)
        
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Transpose the gradient
            transposed_grad = out.grad.transpose(inv_axes)
            
            # Update the gradient of the current tensor
            x.grad = x.grad + transposed_grad if x.grad is not None else transposed_grad
                
    # Store the backward function with respect to the transpose operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x}
    
    # Return the output tensor
    return out


def masked_fill(x: 'Tensor', mask: Union[np.ndarray, 'Tensor'], value: float) -> 'Tensor':
    """
    Fill the masked elements of the tensor with the specified value.
    
    Parameters:
    - x (Tensor): Input tensor
    - mask (np.ndarray): Mask to identify the elements to fill
    - value (float): Value to fill the masked elements
    
    Returns:
    - Tensor: Tensor with the masked elements filled with the specified value
    
    Raises:
    - AssertionError: If the input is not a tensor
    - AssertionError: If the mask is not a numpy array or a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    assert isinstance(mask, (np.ndarray, Tensor)), "Mask must be a numpy array or a tensor"
    
    # Ensure the mask is in numpy format
    mask = mask.data if not isinstance(mask, np.ndarray) else mask
    
    # Fill the tensor with the value where the mask is False
    out_data = np.where(mask, x.data, value)
    
    # Create a new tensor with the filled data
    out = Tensor(out_data, requires_grad=x.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad:
            # Compute the gradient of the loss with respect to the current tensor
            grad_mask = np.where(mask, out.grad if out.grad is not None else 0, 0)
            
            # Update the gradient of the current tensor
            x.grad = x.grad + grad_mask if x.grad is not None else grad_mask
    
    # Store the backward function with respect to the masked fill operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x}
    
    # Return the output tensor
    return out


def clip(x: 'Tensor', min_value: float, max_value: float) -> 'Tensor':
    """
    Applies np.clip to the tensor, limiting its values between min_value and max_value.

    During the backward pass, gradients are propagated only for the elements that are
    within the interval [min_value, max_value]. For clipped elements, the gradient is zero.

    Parameters:
    - min_value (float): The minimum allowed value.
    - max_value (float): The maximum allowed value.

    Returns:
    - Tensor: A new Tensor with values clipped to the range [min_value, max_value]
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Compute forward pass using np.clip.
    out = Tensor(np.clip(x.data, min_value, max_value), requires_grad=x.requires_grad)

    # Define the backward function.
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient.
        if x.requires_grad and out.grad is not None:
            # Create a mask: 1 where self.data is within [min_value, max_value], 0 otherwise.
            mask = ((x.data >= min_value) & (x.data <= max_value)).astype(x.data.dtype)
            
            # Propagate the upstream gradient only where the mask is 1.
            grad_self = mask * out.grad
            
            # Update the gradient of the input tensor.
            x.grad = x.grad + grad_self if x.grad is not None else grad_self

    # Store the backward function with respect to the clip operation.
    out._backward = _backward
    
    # Store the previous tensors in the computation graph.
    out._prev = {x}
    
    # Return the output tensor.
    return out


def gather(x: 'Tensor', indices: 'Tensor', axis: int = 0) -> 'Tensor':
    """
    Gathers values along an axis specified by indices.
    
    Parameters:
    - x (Tensor): Input tensor
    - indices (Tensor): Indices to gather along the axis
    - axis (int): Axis along which to gather the values
    
    Returns:
    - Tensor: Gathered tensor
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Compute the gathered tensor
    out = Tensor(np.take_along_axis(x.data, indices.data.astype(int), axis=axis), requires_grad=x.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Create an array of zeros with the same shape as self.data.
            grad_self = np.zeros_like(x.data)
            
            # Scatter the gradient from out.grad back to grad_self at the positions specified by indices.
            idx = []
            for i in range(x.data.ndim):
                # If the current dimension is the axis, use the indices to gather the gradient.
                if i == axis:
                    # Append the indices to the idx list.
                    idx.append(indices.data.astype(int))
                else:
                    # For other dimensions, create an array of indices.
                    shape = [1] * x.data.ndim
                    shape[i] = x.data.shape[i]
                    
                    # Append the array of indices to the idx list.
                    idx.append(np.arange(x.data.shape[i]).reshape(shape))
                    
            # Use np.add.at to scatter the gradients.
            np.add.at(grad_self, tuple(idx), out.grad)
            
            # Update the gradient of the input tensor
            x.grad = x.grad + grad_self if x.grad is not None else grad_self
            
    # Store the backward function with respect to the gather operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x}
    
    # Return the output tensor
    return out


def unsqueeze(x: 'Tensor', axis: int) -> 'Tensor':
    """
    Unsqueeze the tensor along the specified axis.
    
    Parameters:
    - x (Tensor): Input tensor
    - axis (int): Axis along which to unsqueeze the tensor
    
    Returns:
    - Tensor: Unsqueeze tensor
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Unsqueeze the tensor along the specified axis
    out = Tensor(np.expand_dims(x.data, axis=axis), requires_grad=x.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Squeeze the gradient along the same axis to match the original shape.
            grad_unsqueezed = np.squeeze(out.grad, axis=axis)
            
            # Update the gradient of the input tensor
            x.grad = x.grad + grad_unsqueezed if x.grad is not None else grad_unsqueezed
            
    # Store the backward function with respect to the unsqueeze operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x}
    
    # Return the output tensor
    return out


def concat(tensors: List['Tensor'], axis: int = 0) -> 'Tensor':
    """
    Concatenate a list of tensors along the specified axis.
    
    Parameters:
    - tensors (List[Tensor]): List of input tensors
    - axis (int): Axis along which to concatenate the tensors
    
    Returns:
    - Tensor: Concatenated tensor
    
    Raises:
    - AssertionError: If the the inputs are not tensors
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert all(isinstance(t, Tensor) for t in tensors), "All inputs must be tensors"
    
    # Compute the output tensor by concatenating the input tensors
    # The output tensor requires grad if any of the inputs require grad.
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(np.concatenate([t.data for t in tensors], axis=axis), requires_grad=requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the output tensor has no gradient, return
        if out.grad is None:
            return
        
        # Determine the sizes along the concatenation axis for each tensor
        sizes = [t.data.shape[axis] for t in tensors]
        
        # Compute cumulative indices (start and end indices for each slice)
        indices = np.cumsum([0] + sizes)
        
        # For each tensor, extract the corresponding slice of the gradient
        for t, start, end in zip(tensors, indices[:-1], indices[1:]):
            # If the tensor requires grad, accumulate the gradient
            if t.requires_grad:
                # Create a slicing tuple that selects all elements in each axis,
                # except for the concatenation axis where we slice from start to end.
                slicer = [slice(None)] * out.grad.ndim
                slicer[axis] = slice(start, end)
                
                # Extract the gradient for the current tensor
                grad_piece = out.grad[tuple(slicer)]
                
                # Accumulate the gradient in the corresponding tensor
                t.grad = t.grad + grad_piece if t.grad is not None else grad_piece

    # Store the backward function with respect to the concatenation operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = set(tensors)
    
    # Return the output tensor
    return out


def reshape(x: 'Tensor', new_shape: Tuple[int, ...]) -> 'Tensor':
    """
    Reshape the tensor to the specified new shape.
    
    Parameters:
    - x (Tensor): Input tensor.
    - new_shape (Tuple[int, ...]): The desired shape.
    
    Returns:
    - Tensor: A new Tensor with the specified shape.
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Compute the reshaped tensor
    out = Tensor(x.data.reshape(new_shape), requires_grad=x.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Reshape the gradient from the output back to the shape of x.data
            grad_back = out.grad.reshape(x.data.shape)
            
            # Accumulate the gradient in x.grad
            x.grad = x.grad + grad_back if x.grad is not None else grad_back
            
    # Store the backward function with respect to the reshape operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x}
    
    # Return the output tensor
    return out