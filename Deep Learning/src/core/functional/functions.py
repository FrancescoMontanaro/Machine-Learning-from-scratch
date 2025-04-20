import numpy as np
from typing import Optional, Tuple, List, Union, Type, TYPE_CHECKING, cast

from .utils import accumulate_gradient
if TYPE_CHECKING: from ..tensor import Tensor
from ..utils.context_manager import _NO_GRAD
from ..utils.types_registry import get_tensor_class

from .kernel.max_pool_2d import max_pool_2d_forward, max_pool_2d_gradient
from .kernel.conv_2d import conv_2d_forward, conv_2d_gradient_w, conv_2d_gradient_x


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
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out

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
            accumulate_gradient(x, out.grad)
            
    # Store the backward function with respect to the sum operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    # Return the output tensor
    return out


def max(x: 'Tensor', axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
    """
    Compute the maximum value of the tensor along the specified axis.
    
    Parameters:
    - x (Tensor): Input tensor
    - axis (Optional[Union[int, Tuple[int, ...]]]): Axis along which to compute the maximum value
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
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
            
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Create a mask to identify the maximum elements
            expanded_shape = None
            if axis is not None and not keepdims:
                # Determine the axes as a list.
                axes_list = [axis] if isinstance(axis, int) else list(axis)
                
                # Create the expanded shape from x.data.shape: set each reduction axis to 1.
                expanded_shape = list(x.data.shape)
                for ax in axes_list:
                    expanded_shape[ax] = 1
                
                # Reshape out.data and then broadcast to x.data.shape.
                broadcasted_max = np.broadcast_to(np.reshape(out.data, tuple(expanded_shape)), x.data.shape)
            else:
                # Otherwise, out.data can be directly broadcast.
                broadcasted_max = np.broadcast_to(out.data, x.data.shape)
                
            # Create a mask where x.data equals the broadcasted maximum.
            mask = (x.data == broadcasted_max).astype(x.data.dtype)
            
            # Count how many times the maximum appears along the reduced axes.
            count = np.sum(mask, axis=axis, keepdims=True) if axis is not None else np.sum(mask)
            
            # Distribute the gradient to the maximum elements
            if keepdims or axis is None:
                # If keepdims is True or axis is None, the gradient directly can be broadcasted directly
                grad_x = mask * (np.broadcast_to(out.grad, x.data.shape) / count)
            else:
                # If keepdims is False, it must be expanded along the reduction axes to match the original shape
                if expanded_shape is not None:
                    out_grad_expanded = np.reshape(out.grad, tuple(expanded_shape))
                else:
                    # expanded_shape should not be None if keepdims is False
                    raise ValueError("expanded_shape cannot be None")
                
                # Compute the gradient for each element
                grad_x = mask * (np.broadcast_to(out_grad_expanded, x.data.shape) / count)
            
            # Update the gradient of the input tensor
            accumulate_gradient(x, grad_x)

    # Store the backward function with respect to the maximum operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
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
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Avoid division by zero; assume x.data is non-negative.
            grad_self = out.grad / (2 * np.sqrt(x.data))
            
            # Update the gradient of the input tensor
            accumulate_gradient(x, grad_self)

    # Store the backward function with respect to the square root operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
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
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out

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
            accumulate_gradient(x, grad_self)

    # Store the backward function with respect to the mean operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    # Return the output tensor
    return out


def var(x: 'Tensor', axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, ddof: int = 1) -> 'Tensor':
    """
    Compute the variance of the tensor along the specified axis (or axes).

    Parameters:
    - x (Tensor): Input tensor.
    - axis (Optional[Union[int, Tuple[int, ...]]]): Axis or axes along which to compute the variance.
    - keepdims (bool): Whether to keep the dimensions of the result.
    - ddof (int): Delta degrees of freedom. The divisor used in the calculation is N - ddof, where N represents the number of elements.

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
    square_diff = diff * diff
    
    # Compute the variance by taking the mean of the squared difference.
    var = mean(square_diff, axis=axis, keepdims=keepdims)
    
    # Determine the number of elements over which the variance is computed.
    if axis is None:
        # If axis is None, the variance is computed over all elements.
        num_elements = x.data.size
        
    # If axis is an integer, the variance is computed over the elements in that axis.
    elif isinstance(axis, int):
        # If axis is an integer, the variance is computed over the elements in that axis.
        num_elements = x.data.shape[axis]
        
    # If axis is a tuple, the variance is computed over the elements in the specified axes.
    elif isinstance(axis, tuple):
        # If axis is a tuple, the variance is computed over the elements in the specified axes.
        num_elements = 1
        for ax in axis:
            num_elements *= x.data.shape[ax]
    else:
        raise ValueError("axis must be an integer, a tuple of integers, or None")
    
    # If num_elements > ddof, apply the bessel correction to the variance.
    if ddof != 0 and num_elements > ddof:
        # Convert the population variance to the sample variance.
        var = var * (num_elements / (num_elements - 1))
    
    # Return the variance tensor
    return var


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
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad:
            # Comute the gradient of the loss with respect to the current tensor
            grad = out.data * out.grad
            
            # Compute the gradient of the loss with respect to the current tensor
            accumulate_gradient(x, grad)
            
    # Store the backward function with respect to the exponential operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
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
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Compute the gradient of the loss with respect to the current tensor
            gard = out.grad / x.data
            
            # Update the gradient of the input tensor
            accumulate_gradient(x, gard)
    
    # Store the backward function with respect to the natural logarithm operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
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
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # Invert the axes
        inv_axes = np.argsort(axes)
        
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Transpose the gradient
            transposed_grad = out.grad.transpose(inv_axes)
            
            # Update the gradient of the current tensor
            accumulate_gradient(x, transposed_grad)
                
    # Store the backward function with respect to the transpose operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
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
    out_data = np.where(mask, value, x.data)
    
    # Create a new tensor with the filled data
    out = Tensor(out_data, requires_grad=x.requires_grad)
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad:
            # Compute the gradient of the loss with respect to the current tensor
            grad_mask = np.where(mask, 0, out.grad if out.grad is not None else 0)
            
            # Update the gradient of the current tensor
            accumulate_gradient(x, grad_mask)
    
    # Store the backward function with respect to the masked fill operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
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
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out

    # Define the backward function.
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient.
        if x.requires_grad and out.grad is not None:
            # Create a mask: 1 where self.data is within [min_value, max_value], 0 otherwise.
            mask = ((x.data >= min_value) & (x.data <= max_value)).astype(x.data.dtype)
            
            # Propagate the upstream gradient only where the mask is 1.
            grad_self = mask * out.grad
            
            # Update the gradient of the input tensor.
            accumulate_gradient(x, grad_self)

    # Store the backward function with respect to the clip operation.
    out._backward = _backward
    
    # Store the previous tensors in the computation graph.
    out._prev = {x} if x.requires_grad else set()
    
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
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
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
            accumulate_gradient(x, grad_self)
            
    # Store the backward function with respect to the gather operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    # Return the output tensor
    return out


def squeeze(x: 'Tensor', axis: Optional[int] = None) -> 'Tensor':
    """
    Squeeze the tensor by removing singleton dimensions along the specified axis.
    
    Parameters:
    - x (Tensor): Input tensor
    - axis (int or None): Axis along which to squeeze the tensor. If None, all singleton dimensions are removed.
    
    Returns:
    - Tensor: Squeezed tensor
    
    Raises:
    - AssertionError: If the input is not a tensor
    - ValueError: If the specified axis is not a singleton dimension
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Squeeze the tensor along the specified axis
    out = Tensor(np.squeeze(x.data, axis=axis), requires_grad=x.requires_grad)
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Unsqueeze the gradient along the same axis to match the original shape
            if axis is None:
                # For None case, we need to restore all squeezed dims
                grad_squeezed = out.grad
                original_shape = x.data.shape
                for dim in sorted([i for i, size in enumerate(original_shape) if size == 1], reverse=True):
                    grad_squeezed = np.expand_dims(grad_squeezed, axis=dim)
            else:
                # For specific axis case
                grad_squeezed = np.expand_dims(out.grad, axis=axis)
            
            # Update the gradient of the input tensor
            accumulate_gradient(x, grad_squeezed)
            
    # Store the backward function with respect to the squeeze operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    # Return the output tensor
    return out


def unsqueeze(x: 'Tensor', axis: int) -> 'Tensor':
    """
    Unsqueeze the tensor along the specified axis.
    
    Parameters:
    - x (Tensor): Input tensor
    - axis (int): Axis along which to unsqueeze the tensor
    
    Returns:
    - Tensor: Unsqueezed tensor
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Unsqueeze the tensor along the specified axis
    out = Tensor(np.expand_dims(x.data, axis=axis), requires_grad=x.requires_grad)
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Squeeze the gradient along the same axis to match the original shape.
            grad_unsqueezed = np.squeeze(out.grad, axis=axis)
            
            # Update the gradient of the input tensor
            accumulate_gradient(x, grad_unsqueezed)
            
    # Store the backward function with respect to the unsqueeze operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
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
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
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
                accumulate_gradient(t, grad_piece)

    # Store the backward function with respect to the concatenation operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {t for t in tensors if t.requires_grad}
    
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
    
    Raises:
    - AssertionError: If the input is not a tensor.
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Compute the reshaped tensor
    out = Tensor(x.data.reshape(new_shape), requires_grad=x.requires_grad)
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Reshape the gradient from the output back to the shape of x.data
            grad_back = out.grad.reshape(x.data.shape)
            
            # Accumulate the gradient in x.grad
            accumulate_gradient(x, grad_back)
            
    # Store the backward function with respect to the reshape operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    # Return the output tensor
    return out


def repeat(x: 'Tensor', repeats: int, axis: Optional[int] = None) -> 'Tensor':
    """
    Repeat elements of a tensor along a specified axis.
    
    Parameters:
    - x (Tensor): Input tensor.
    - repeats (int): Number of repetitions for each element.
    - axis (Optional[int]): The axis along which to repeat. If None, the array is flattened before repeating, and the output will be 1D.
    
    Returns:
    - Tensor: A new tensor with repeated elements.
    
    Raises:
    - AssertionError: If the input is not a tensor.
    """
    
    # Get the Tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Compute the output tensor by repeating the input tensor
    out = Tensor(np.repeat(x.data, repeats, axis=axis), requires_grad=x.requires_grad)
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # If axis is None, np.repeat flattens x, so the gradient must be reshaped back to the original shape
            if axis is None:
                # out.grad has shape (x.data.size, * repeats,). Reshape to (x.data.size, repeats) then sum along the repeated axis.
                grad_unrepeated = out.grad.reshape(x.data.size, repeats).sum(axis=1)
                
                # Finally, reshape back to the original shape of x.data
                grad_unrepeated = grad_unrepeated.reshape(x.data.shape)
            else:
                # Insert the repeats dimension into the shape along the specified axis:
                new_shape = (
                    x.data.shape[:axis] +
                    (x.data.shape[axis], repeats) +
                    x.data.shape[axis+1:]
                )
                
                # Sum along the repeated axis to get the gradient for each element
                grad_unrepeated = out.grad.reshape(new_shape).sum(axis=axis+1)
            
            # Accumulate the unrepeated gradient in x.grad
            accumulate_gradient(x, grad_unrepeated)

    # Store the backward function with respect to the repeat operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    # Return the output tensor
    return out


def pad(x: 'Tensor', pad_width: Tuple[Tuple[int, int], ...]) -> 'Tensor':
    """
    Pad the tensor with zeros or a constant value.

    Parameters:
    - x (Tensor): Input tensor.
    - pad_width (Tuple[Tuple[int, int], ...]): Tuple of pad widths for each dimension. For example, for a 2D tensor, ((pad_top, pad_bottom), (pad_left, pad_right)).
    - mode (str): Padding mode (default "constant").

    Returns:
    - Tensor: A new tensor with padded data.
    
    Raises:
    - AssertionError: If the input is not a tensor.
    """
    
    # Get the Tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Compute the padded tensor
    out = Tensor(np.pad(x.data, pad_width, mode="constant"), requires_grad=x.requires_grad)
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # For each axis, create a slice from pad_width[axis][0] to pad_width[axis][0] + original_dim.
            slices = tuple(slice(pw[0], pw[0] + s) for s, pw in zip(x.data.shape, pad_width))
            
            # Extract the portion of out.grad corresponding to the original tensor's shape.
            grad_unpadded = out.grad[slices]
            
            # Update the gradient of the input tensor
            accumulate_gradient(x, grad_unpadded)
    
    # Store the backward function with respect to the pad operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    # Return the output tensor
    return out


def conv_2d(x: 'Tensor', kernel: 'Tensor', stride: Tuple[int, int] = (1,1)) -> 'Tensor':
    """
    Compute the 2D convolution of the input tensor with the kernel.
    
    Parameters:
    - x (Tensor): Input tensor.
    - kernel (Tensor): Convolution kernel.
    - stride (Tuple[int, int]): Stride of the convolution. Default is (1, 1).
    
    Returns:
    - Tensor: The result of the 2D convolution.
    
    Raises:
    - AssertionError: If the inputs are not tensors.
    - AssertionError: If the input channels do not match the kernel channels.
    - ValueError: If the kernel or stride is too large for the input size.
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Ensure the inputs are tensors
    assert isinstance(x, Tensor) and isinstance(kernel, Tensor), "Both inputs must be tensors"
    
    # Extract the input dimensions
    batch_size, height, width, channels = x.shape()
    out_channels, kernel_in_channels, kernel_height, kernel_width = kernel.shape()
    stride_height, stride_width = stride
    
    # Check if the input channels match the kernel channels
    assert kernel_in_channels == channels, "w in_channels != x channels"
    
    # Compute the output dimensions
    out_height = (height - kernel_height) // stride_height + 1
    out_width = (width - kernel_width) // stride_width + 1
    
    # Check if the kernel is too large or the stride is too large for the input size
    if out_height < 1 or out_width < 1:
        raise ValueError("Kernel or stride too large for input size")

    # Create the output array
    out_data = np.empty((batch_size, out_height, out_width, out_channels), dtype=x.data.dtype)
    
    # Perform the 2D convolution
    conv_2d_forward(x.data, kernel.data, stride_height, stride_width, out_data)
    
    # Create the output tensor
    out = Tensor(out_data, requires_grad=x.requires_grad or kernel.requires_grad)
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward():
        # Check if the output gradient is None, and return if so
        if out.grad is None:
            return
        
        # Gradient for kernel
        if kernel.requires_grad:
            # Check if kernel gradient is None, and initialize it if so
            if kernel.grad is None:
                kernel.grad = np.zeros_like(kernel.data)
                
            # Compute the gradient with respect to the kernel
            conv_2d_gradient_w(x.data, out.grad, stride_height, stride_width, kernel.grad)
            
        # Gradient for input
        if x.requires_grad:
            # Check if input gradient is None, and initialize it if so
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
                
            # Compute the gradient with respect to the input
            conv_2d_gradient_x(out.grad, kernel.data, stride_height, stride_width, x.grad)
    
    # Store the backward function with respect to the convolution operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {t for t in (x, kernel) if t.requires_grad}
    
    # Return the output tensor
    return out


def max_pool_2d(x: 'Tensor', kernel_size: Tuple[int,int] = (2,2), stride: Tuple[int,int] = (2,2)) -> 'Tensor':
    """
    Apply a 2D Max Pooling over an input tensor of shape (N, H, W, C).
    
    Parameters:
    - x (Tensor): The input tensor, shape (N, H, W, C).
    - kernel_size (Tuple[int,int]): The spatial size of the window (kH, kW).
    - stride (Tuple[int,int]): The stride (stride_height, stride_width).
    
    Returns:
    - Tensor: The output after applying Max Pool 2D, shape (N, outH, outW, C).
    
    Raises:
    - AssertionError: If x is not a Tensor.
    - ValueError: If the window or stride is too large for the input size.
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input x must be a Tensor."
    
    # Extract the input dimensions
    batch_size, height, width, channels = x.shape()
    kernel_height, kernel_width = kernel_size
    stride_height, stride_width = stride
    
    # Compute the output dimensions
    out_height = (height - kernel_height) // stride_height + 1
    out_width = (width - kernel_width) // stride_width + 1
    
    # Check if the kernel or stride is too large for the input size
    if out_height < 1 or out_width < 1:
        raise ValueError("Kernel size or stride too large for input size.")

    # Initialize the output array and the indices for max pooling
    out_data = np.empty((batch_size, out_height, out_width, channels), dtype=x.data.dtype)
    arg_i = np.zeros_like(out_data, dtype=np.int32)
    arg_j = np.zeros_like(out_data, dtype=np.int32)

    # Perform the max pooling operation
    max_pool_2d_forward(x.data, kernel_height, kernel_width, stride_height, stride_width, out_data, arg_i, arg_j)
    
    # Compute the max pooling operation and store the output tensor
    out = Tensor(out_data, requires_grad=x.requires_grad)
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward():
        # Check if the output gradient is None, and return if so
        if out.grad is None:
            return
        
        # Check if the input tensor requires gradient
        if x.grad is None:
            # Initialize the gradient of x if it is None
            x.grad = np.zeros_like(x.data)
            
        # Backprop the gradient through the max pooling operation
        max_pool_2d_gradient(arg_i, arg_j, out.grad, stride_height, stride_width, x.grad)
    
    # Store the backward function with respect to the max pooling operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    # Return the output tensor
    return out