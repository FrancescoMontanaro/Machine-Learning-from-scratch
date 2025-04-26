import numpy as np
from typing import Optional, Tuple, List, Union, Type, TYPE_CHECKING, cast

from .utils import accumulate_gradient
if TYPE_CHECKING: from ..tensor import Tensor
from ..utils.context_manager import _NO_GRAD
from ..utils.types_registry import get_tensor_class

# Importing the kernel functions
from .kernel.exp import exp_gradient
from .kernel.log import log_gradient
from .kernel.sqrt import sqrt_gradient
from .kernel.mean import mean_flat_backward
from .kernel.pad import pad_forward, pad_gradient
from .kernel.repeat import repeat_forward, repeat_gradient
from .kernel.sum import sum_flat_forward, sum_flat_gradient
from .kernel.max import max_flat_forward, max_flat_gradient
from .kernel.var import var_flat_forward, var_flat_gradient
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
    
    # forward
    if axis is None:
        # Create a buffer to store the sum
        buf = np.zeros((1,), dtype=x.data.dtype)
        
        # Compute the sum of the flattened tensor
        sum_flat_forward(x.data.ravel(), buf)
        
        # If keepdims is True, create an output tensor with the same shape as the input tensor
        if keepdims:
            # Create an output tensor with the same shape as the input tensor
            out_data = np.full([1]*x.data.ndim, buf[0], dtype=x.data.dtype)
        else:
            # Create an output tensor with the shape of the sum
            out_data = buf[0]
    else:
        # If axis is not None, compute the sum along the specified axis
        out_data = x.data.sum(axis=axis, keepdims=keepdims)
    
    # Compute the sum of the tensor along the specified axis
    out = Tensor(out_data, requires_grad=x.requires_grad)
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out

    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # If the gradient is None, create a zero gradient tensor
            if x.grad is None:
                # Create a zero gradient tensor with the same shape as the input tensor
                x.grad = np.zeros_like(x.data)
                
            # If axis is None, broadcast the gradient to the shape of the input tensor
            if axis is None:
                # Broadcast the gradient to the shape of the input tensor
                sum_flat_gradient(np.array([out.grad]).ravel(), x.grad.ravel())
            else:
                # If axis is not None, compute the gradient along the specified axis
                grad = out.grad
                
                # If axis is a tuple, compute the gradient for each axis in the tuple
                if not keepdims:
                    # If keepdims is False, expand the gradient along the specified axis
                    grad = np.expand_dims(grad, axis=axis)
                    
                # Accumulate the gradient in the input tensor
                accumulate_gradient(x, np.broadcast_to(grad, x.data.shape))
            
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
    
    # Initialize the index variable
    idx = None
    
    # If axis is None, compute the maximum value of the flattened tensor
    if axis is None:
        # Create a buffer to store the maximum value and its index
        buf = np.zeros((1,), dtype=x.data.dtype)
        idx = np.zeros((1,), dtype=np.int64)
        
        # Compute the maximum value of the flattened tensor
        max_flat_forward(x.data.ravel(), buf, idx)
        
        # If keepdims is True, create an output tensor with the same shape as the input tensor
        if keepdims:
            # Create an output tensor with the same shape as the input tensor
            out_data = np.full([1]*x.data.ndim, buf[0], dtype=x.data.dtype)
        # If keepdims is False, create an output tensor with the shape of the maximum value
        else:
            # Create an output tensor with the shape of the maximum value
            out_data = buf[0]
    # If axis is not None, compute the maximum value along the specified axis
    else:
        # If axis is not None, compute the maximum value along the specified axis
        out_data = np.max(x.data, axis=axis, keepdims=keepdims)
    
    # Compute the maximum value of the tensor along the specified axis
    out = Tensor(out_data, requires_grad=x.requires_grad)
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
            
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # If the gradient is None, create a zero gradient tensor
            if x.grad is None:
                # Create a zero gradient tensor with the same shape as the input tensor
                x.grad = np.zeros_like(x.data)
                
            # If axis is None, compute the gradient for the flattened tensor
            if axis is None:
                # Compute the index of the maximum value
                max_flat_gradient(idx, np.array([out.grad]).ravel(), x.grad.ravel())
                
            # If axis is not None, compute the gradient along the specified axis
            else:
                # Initialize the gradient tensor
                expanded = out.grad
                
                # If axis is a tuple, expand the gradient for each axis in the tuple
                if not keepdims:
                    # If keepdims is False, expand the gradient along the specified axis
                    expanded = np.expand_dims(expanded, axis=axis)
                    
                # Create a mask to identify the maximum values
                mask = (x.data == expanded)

                # Count the number of maximum values along the specified axis
                count = np.sum(mask, axis=axis, keepdims=True)
                
                # Avoid division by zero; set count to 1 where it is zero
                grad_x = mask * (expanded / count)
                
                # Accumulate the gradient in the input tensor
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
            # If the gradient is None, create a zero gradient tensor
            if x.grad is None:
                # Create a zero gradient tensor with the same shape as the input tensor
                x.grad = np.zeros_like(x.data)
                
            # Compute the gradient of the square root operation
            sqrt_gradient(out.grad.ravel(), x.data.ravel(), x.grad.ravel())

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
    - TypeError: If the axis is not an integer or a tuple of integers
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
            # If the gradient is None, create a zero gradient tensor
            if x.grad is None:
                # Create a zero gradient tensor with the same shape as the input tensor
                x.grad = np.zeros_like(x.data)
                
            # If axis is None, compute the gradient for the flattened tensor
            if axis is None:
                # Create a buffer to store the gradient
                buf = np.zeros((1,), dtype=x.data.dtype)
                
                # Initialize the buffer with the gradient of the output tensor
                buf[0] = out.grad
                inv = 1.0 / x.data.size
                
                # Compute the gradient of the mean operation
                mean_flat_backward(buf, x.grad.ravel(), inv)
                
            # If axis is not None, compute the gradient along the specified axis
            else:
                # Initialize the gradient tensor
                grad = out.grad
                
                # If keepdims is False, expand the gradient for each axis in the tuple
                if not keepdims:
                    # Expand the gradient along the specified axis
                    grad = np.expand_dims(grad, axis=axis)

                # Compute the number of elements along the specified axis/axes
                if isinstance(axis, int):
                    # If axis is an integer, compute the number of elements along that axis
                    num_elements_along_axis = x.data.shape[axis]
                elif isinstance(axis, tuple):
                    # If axis is a tuple, compute the number of elements along each axis
                    num_elements_along_axis = np.prod([x.data.shape[ax] for ax in axis])
                else:
                    # If axis is not an integer or a tuple, raise a TypeError
                    raise TypeError("axis must be an int or a tuple of ints")

                # Accumulate the gradient in the input tensor
                accumulate_gradient(x, grad / num_elements_along_axis)

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
    
    # If axis is None, compute the variance of the flattened tensor
    if axis is None:
        # Create a buffer to store the variance
        buf = np.zeros((1,), dtype=x.data.dtype)
        
        # Compute the variance of the flattened tensor
        var_flat_forward(x.data.ravel(), buf, ddof)
        
        # If keepdims is True, create an output tensor with the same shape as the input tensor
        if keepdims:
            # Create an output tensor with the same shape as the input tensor
            out_data = np.full([1]*x.data.ndim, buf[0], dtype=x.data.dtype)
            
        # If keepdims is False, create an output tensor with the shape of the variance
        else:
            # Create an output tensor with the shape of the variance
            out_data = buf[0]
    
    # If axis is not None, compute the variance along the specified axis
    else:
        # If axis is not None, compute the variance along the specified axis
        out_data = np.var(x.data, axis=axis, keepdims=keepdims, ddof=ddof)
        
    # Create the output tensor with the computed variance
    out = Tensor(out_data, requires_grad=x.requires_grad)
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD:
        return out
    
    # Define the backward function
    def _backward():
        # Check if the gradient needs to be computed
        if x.requires_grad and out.grad is not None:
            # If the gradient is None, create a zero gradient tensor
            if x.grad is None:
                # Create a zero gradient tensor with the same shape as the input tensor
                x.grad = np.zeros_like(x.data)
                
            # If axis is None, compute the gradient for the flattened tensor
            if axis is None:
                # Compute the gradient of the variance operation
                var_flat_gradient(np.array([out.grad]).ravel(), x.data.ravel(), x.grad.ravel(), ddof)
                
            # If axis is not None, compute the gradient along the specified axis
            else:
                # Initialize the gradient tensor
                grad = out.grad
                
                # If keepdims is False, expand the gradient for each axis in the tuple
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                    
                # Broadcast the gradient to the shape of the input tensor
                factor = 1.0 / (x.data.shape[axis] - ddof) if isinstance(axis,int) and ddof and x.data.shape[axis]>ddof else 1.0 / x.data.size
                
                # Compute the number of elements along the specified axis/axes
                grad_x = (x.data - np.mean(x.data, axis=axis, keepdims=True)) * 2 * factor * grad
                
                # Accumulate the gradient in the input tensor
                accumulate_gradient(x, grad_x)
                
    # Store the backward function with respect to the variance operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    # Return the output tensor
    return out


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
        if x.requires_grad and out.grad is not None:
            # If the gradient is None, create a zero gradient tensor
            if x.grad is None:
                # Create a zero gradient tensor with the same shape as the input tensor
                x.grad = np.zeros_like(x.data)
                
            # Compute the gradient of the exponential operation
            exp_gradient(out.data.ravel(), out.grad.ravel(), x.grad.ravel())
            
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
            # If the gradient is None, create a zero gradient tensor
            if x.grad is None:
                # Create a zero gradient tensor with the same shape as the input tensor
                x.grad = np.zeros_like(x.data)
                
            # Compute the gradient of the logarithm operation
            log_gradient(x.data.ravel(), out.grad.ravel(), x.grad.ravel())
    
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
            # Invert the axes to match the original tensor
            inv_axes = np.argsort(axes)
            
            # Transpose the gradient to match the original tensor
            grad_x = out.grad.transpose(inv_axes)
            
            # Accumulate the gradient in the input tensor
            accumulate_gradient(x, grad_x)
                
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
    
    # If the axis is None, flatten the tensor and repeat
    if axis is None:
        # Flatten the tensor
        out_data = np.empty(x.data.size * repeats, dtype=x.data.dtype)
        
        # Repeat the flattened tensor
        repeat_forward(x.data.ravel(), repeats, out_data)
    # If the axis is specified, repeat along that axis
    else:
        # Repeat the tensor along the specified axis
        out_data = np.repeat(x.data, repeats, axis=axis)
    
    # Compute the output tensor by repeating the input tensor
    out = Tensor(out_data, requires_grad=x.requires_grad)
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Check if axis is None
            if axis is None:
                # Check if the gradient is None
                if x.grad is None:
                    # Initialize the gradient to zeros
                    x.grad = np.zeros_like(x.data)
                    
                # Repeat the gradient along the specified axis
                repeat_gradient(out.grad.ravel(), repeats, x.grad.ravel())
            
            # If axis is specified, repeat the gradient along that axis
            else:
                # Reduce the gradient along the specified axis
                grad_unrepeated = np.add.reduce(
                    out.grad.reshape(
                        *(x.data.shape[:axis]), x.data.shape[axis], repeats, *x.data.shape[axis+1:]
                    ), axis=axis+1
                )
                
                # Accumulate the gradient
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
    
    # Extract the padding widths for each dimension and the input tensor shape
    (_, _), (pt, pb), (pl, pr), (_, _) = pad_width
    batch_size, height, width, channels = x.shape()
    
    # Create the output tensor with the new shape
    out_data = np.empty((batch_size, height + pt + pb, width + pl + pr, channels), dtype=x.data.dtype)
    
    # Perform the padding operation
    pad_forward(x.data, pt, pb, pl, pr, out_data)
    
    # Compute the padded tensor
    out = Tensor(out_data, requires_grad=x.requires_grad)
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Check if the gradient is None
            if x.grad is None:
                # Initialize the gradient to zeros
                x.grad = np.zeros_like(x.data)
                
            # Compute the gradient with respect to the input tensor
            pad_gradient(out.grad, pt, pb, pl, pr, x.grad)
    
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