import numpy as np
from typing import Optional, Tuple, List, Union, TYPE_CHECKING

if TYPE_CHECKING: from ..tensor import Tensor
from .base import tensor_unary_op, tensor_binary_op, tensor_nary_op, accumulate_gradient

# Importing the kernel functions
from .kernel.exp import exp_gradient
from .kernel.log import log_gradient
from .kernel.sqrt import sqrt_gradient
from .kernel.mean import mean_flat_backward
from .kernel.squeeze import squeeze_gradient
from .kernel.unsqueeze import unsqueeze_gradient
from .kernel.pad import pad_forward, pad_gradient
from .kernel.clip import clip_forward, clip_gradient
from .kernel.concat import concat_forward, concat_gradient
from .kernel.repeat import repeat_forward, repeat_gradient
from .kernel.sum import sum_flat_forward, sum_flat_gradient
from .kernel.max import max_flat_forward, max_flat_gradient
from .kernel.max_pool_2d import max_pool_2d_forward, max_pool_2d_gradient
from .kernel.conv_2d import conv_2d_forward, conv_2d_gradient_w, conv_2d_gradient_x
from .kernel.masked_fill import masked_fill_forward, masked_fill_gradient, masked_fill_forward_neg_inf, masked_fill_forward_inf


def sum(x: 'Tensor', axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
    """
    Compute the sum of the tensor along the specified axis.
    
    Parameters:
    - x (Tensor): Input tensor
    - axis (int): Axis along which to compute the sum
    - keepdims (bool): Whether to keep the dimensions of the input tensor
    
    Returns:
    - Tensor: Sum of the tensor along the specified axis
    """
    
    # Define the forward function
    def forward() -> np.ndarray:
        # If axis is None, compute the sum of the flattened tensor
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
            
        # Return the computed sum
        return out_data

    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the gradient needs to be computed
        if not x.requires_grad: 
            return
        
        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # If axis is None, broadcast the gradient to the shape of the input tensor
        if axis is None:
            # Broadcast the gradient to the shape of the input tensor
            sum_flat_gradient(np.array([out_grad]).ravel(), x.grad.ravel())
        else:
            # If axis is not None, compute the gradient along the specified axis
            grad = out_grad
            
            # If axis is a tuple, compute the gradient for each axis in the tuple
            if not keepdims:
                # If keepdims is False, expand the gradient along the specified axis
                grad = np.expand_dims(grad, axis=axis)
                
            # Accumulate the gradient in the input tensor
            accumulate_gradient(x, np.broadcast_to(grad, x.data.shape))

    # Return the tensor operation withe the specified forward and backward functions         
    return tensor_unary_op(x, forward, backward)


def max(x: 'Tensor', axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
    """
    Compute the maximum value of the tensor along the specified axis.
    
    Parameters:
    - x (Tensor): Input tensor
    - axis (Optional[Union[int, Tuple[int, ...]]]): Axis along which to compute the maximum value
    - keepdims (bool): Whether to keep the dimensions of the input tensor
    
    Returns:
    - Tensor: Maximum value of the tensor along the specified axis
    """
    
    # Create the idx array to store the indices of the maximum values
    idx: np.ndarray
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Set the nonlocal variable idx to store the indices of the maximum values
        nonlocal idx
        
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
            
        # Return the computed maximum value and its index
        return out_data
            
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the tensor requires gradient computation
        if not x.requires_grad:
            return
           
        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # If axis is None, compute the gradient for the flattened tensor
        if axis is None:
            # Compute the index of the maximum value
            max_flat_gradient(idx, np.array([out_grad]).ravel(), x.grad.ravel())
            
        # If axis is not None, compute the gradient along the specified axis
        else:
            # Initialize the gradient tensor
            expanded = out_grad
            
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

    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)


def sqrt(x: 'Tensor') -> 'Tensor':
    """
    Compute the element-wise square root of the tensor.
    
    Parameters:
    - x (Tensor): Input tensor.
    
    Returns:
    - Tensor: Square root of the input tensor.
    """
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Compute the square root of the tensor
        return np.sqrt(x.data)
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the gradient needs to be computed
        if not x.requires_grad: 
            return
        
        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # Compute the gradient of the square root operation
        sqrt_gradient(out_grad.ravel(), x.data.ravel(), x.grad.ravel())
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)


def mean(x: 'Tensor', axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
    """
    Compute the mean of the tensor along the specified axis.
    
    Parameters:
    - axis (Optional[Union[int, Tuple[int, ...]]]): Axis along which to compute the mean
    - keepdims (bool): Whether to keep the dimensions of the input tensor
    
    Returns:
    - Tensor: Mean of the tensor along the specified axis
    """
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Compute the mean of the tensor along the specified axis
        return np.mean(x.data, axis=axis, keepdims=keepdims)
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the gradient needs to be computed
        if not x.requires_grad: 
            return
        
        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # If axis is None, compute the gradient for the flattened tensor
        if axis is None:
            # Create a buffer to store the gradient
            buf = np.zeros((1,), dtype=x.data.dtype)
            
            # Initialize the buffer with the gradient of the output tensor
            buf[0] = out_grad
            inv = 1.0 / x.data.size
            
            # Compute the gradient of the mean operation
            mean_flat_backward(buf, x.grad.ravel(), inv)
            
        # If axis is not None, compute the gradient along the specified axis
        else:
            # Initialize the gradient tensor
            grad = out_grad
            
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

    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)


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
    """
    
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
        num_elements = int(np.prod([x.data.shape[ax] for ax in axis]))
    else:
        # If axis is not an integer or a tuple, raise a TypeError.
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
    """
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Compute the exponential of the tensor
        return np.exp(x.data)
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the gradient needs to be computed
        if not x.requires_grad: 
            return
        
        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # Compute the gradient of the exponential operation
        exp_gradient(out_grad.ravel(), x.data.ravel(), x.grad.ravel())
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)


def log(x: 'Tensor') -> 'Tensor':
    """
    Computes the natural logarithm of the tensor.
    
    Returns:
    - Tensor: Natural logarithm of the tensor
    """
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Compute the natural logarithm of the tensor
        return np.log(x.data)
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the gradient needs to be computed
        if not x.requires_grad: 
            return

        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # Compute the gradient of the logarithm operation
        log_gradient(x.data.ravel(), out_grad.ravel(), x.grad.ravel())
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)


def transpose(x: 'Tensor', axes: Tuple[int]) -> 'Tensor':
    """
    Compute the transpose of the tensor.
    
    Parameters:
    - x (Tensor): Input tensor
    - axes (Tuple[int]): Permutation of the dimensions
    
    Returns:
    - Tensor: Transpose of the tensor
    """
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Compute the transpose of the tensor
        return x.data.transpose(axes)
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the gradient needs to be computed
        if not x.requires_grad: 
            return
        
        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # Invert the axes to match the original tensor
        inv_axes = np.argsort(axes)
        
        # Transpose the gradient to match the original tensor
        grad_x = out_grad.transpose(inv_axes)
        
        # Accumulate the gradient in the input tensor
        accumulate_gradient(x, grad_x)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)


def masked_fill(x: 'Tensor', mask: Union[np.ndarray, 'Tensor'], value: float) -> 'Tensor':
    """
    Fill the masked elements of the tensor with the specified value.
    
    Parameters:
    - x (Tensor): Input tensor
    - mask (np.ndarray): Mask to identify the elements to fill
    - value (float): Value to fill the masked elements
    
    Returns:
    - Tensor: Tensor with the masked elements filled with the specified value
    """
    
    # Define the mask_flat variable to store the flattened mask
    mask_flat: np.ndarray
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Set the nonlocal variable mask_flat to store the flattened mask
        nonlocal mask_flat
        
        # Extract the mask data
        mask_arr = mask if isinstance(mask, np.ndarray) else mask.data
        
        # Flatten the mask and input tensor data
        mask_flat = mask_arr.ravel()
        flat_data = x.data.ravel()
        
        # Prepare the output tensor and flatten it
        out_data = np.empty_like(x.data)
        out_flat = out_data.ravel()
        
        if isinstance(value, float) and not np.isfinite(value):
            # The value is negative infinity
            if value < 0:
                # Perform the masked fill operation with negative infinity
                masked_fill_forward_neg_inf(flat_data, mask_flat, out_flat)
                
            # The value is positive infinity
            else:
                # Perform the masked fill operation with positive infinity
                masked_fill_forward_inf(flat_data, mask_flat, out_flat)
        else:
            # Cast the value to the appropriate type    
            fill_val = x.data.dtype.type(value)
            
            # Perform the masked fill operation
            masked_fill_forward(flat_data, mask_flat, fill_val, out_flat)
    
        # Return the output data
        return out_data
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the gradient needs to be computed
        if not x.requires_grad: 
            return

        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # Backpropagate the gradient through the masked fill operation
        masked_fill_gradient(mask_flat, out_grad.ravel(), x.grad.ravel())
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)


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
    """
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Create an output tensor with the same shape as the input tensor
        out_data = np.empty_like(x.data)
    
        # Clip the values of the tensor to the specified range
        clip_forward(x.data.ravel(), min_value, max_value, out_data.ravel())
        
        # Return the clipped tensor
        return out_data
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the gradient needs to be computed
        if not x.requires_grad: 
            return
        
        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # Compute the gradient of the clipping operation
        clip_gradient(x.data.ravel(), out_grad.ravel(), x.grad.ravel(), min_value, max_value)
            
    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)


def squeeze(x: 'Tensor', axis: Optional[int] = None) -> 'Tensor':
    """
    Squeeze the tensor by removing singleton dimensions along the specified axis.
    
    Parameters:
    - x (Tensor): Input tensor
    - axis (int or None): Axis along which to squeeze the tensor. If None, all singleton dimensions are removed.
    
    Returns:
    - Tensor: Squeezed tensor
    """
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Squeeze the tensor along the specified axis
        return np.squeeze(x.data, axis=axis)
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the gradient needs to be computed
        if not x.requires_grad: 
            return
        
        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # Unsqueeze the gradient along the same axis to match the original shape
        if axis is None:
            # For None case, we need to restore all squeezed dims
            grad_squeezed = out_grad
            original_shape = x.data.shape
            for dim in sorted([i for i, size in enumerate(original_shape) if size == 1], reverse=True):
                grad_squeezed = np.expand_dims(grad_squeezed, axis=dim)
        else:
            # For specific axis case
            grad_squeezed = np.expand_dims(out_grad, axis=axis)
        
        # Update the gradient of the input tensor
        accumulate_gradient(x, grad_squeezed)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)


def unsqueeze(x: 'Tensor', axis: int) -> 'Tensor':
    """
    Unsqueeze the tensor along the specified axis.
    
    Parameters:
    - x (Tensor): Input tensor
    - axis (int): Axis along which to unsqueeze the tensor
    
    Returns:
    - Tensor: Unsqueezed tensor
    """
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Unsqueeze the tensor along the specified axis
        return np.expand_dims(x.data, axis=axis)
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the gradient needs to be computed
        if not x.requires_grad: 
            return
        
        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # Compute the gradient of the unsqueeze operation
        unsqueeze_gradient(out_grad.ravel(), x.grad.ravel())
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)


def concat(tensors: List['Tensor'], axis: int = 0) -> 'Tensor':
    """
    Concatenate a list of tensors along the specified axis.
    
    Parameters:
    - tensors (List[Tensor]): List of input tensors
    - axis (int): Axis along which to concatenate the tensors
    
    Returns:
    - Tensor: Concatenated tensor
    """
    
    # Define the offset array to store the offsets of each tensor
    offsets: np.ndarray
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Set the nonlocal variable offsets to store the offsets of each tensor
        nonlocal offsets
        
        # Flatten the tensors and compute their sizes
        parts = [t.data.ravel() for t in tensors]
        sizes = [p.size for p in parts]
        offsets = np.array(np.cumsum([0] + sizes), dtype=np.int64)

        # Create a flattened array to hold the concatenated tensors
        parts_flat = np.empty(offsets[-1], dtype=tensors[0].data.dtype)
        
        # Interleave the flattened tensors into a single array
        for i, p in enumerate(parts):
            # Fill the flattened array with the data from each tensor
            parts_flat[offsets[i]:offsets[i+1]] = p
            
        # Create an output array to hold the concatenated result
        out_flat = np.empty_like(parts_flat)
        
        # Concatenate the flattened tensors along the specified axis
        concat_forward(parts_flat, offsets, out_flat)
        
        # Extract the shape of the first tensor to determine the output shape
        out_shape = list(tensors[0].data.shape)
        
        # Update the shape of the output tensor along the specified axis
        out_shape[axis] = int(np.sum([t.data.shape[axis] for t in tensors]))
        
        # Reshape the output data to match the concatenated shape
        out_data = out_flat.reshape(out_shape)
        
        # Return the concatenated tensor
        return out_data
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Flatten the output gradient
        flat_grad = out_grad.ravel()
        
        # Iterate over the tensors and their offsets
        for i, t in enumerate(tensors):
            # Check if the tensor requires gradient computation
            if not t.requires_grad:
                continue
            
            # Check if the gradient is initialized
            assert t.grad is not None, f"Gradient of the {i}-th tensor must be initialized"
                
            # Compute the gradient of the concatenation operation
            concat_gradient(offsets, flat_grad, t.grad.ravel(), i)

    # Return the tensor operation with the specified forward and backward functions
    return tensor_nary_op(tensors, forward, backward)


def reshape(x: 'Tensor', new_shape: Tuple[int, ...]) -> 'Tensor':
    """
    Reshape the tensor to the specified new shape.
    
    Parameters:
    - x (Tensor): Input tensor.
    - new_shape (Tuple[int, ...]): The desired shape.
    
    Returns:
    - Tensor: A new Tensor with the specified shape.
    """
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Reshape the tensor to the specified new shape
        return x.data.reshape(new_shape)
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the gradient needs to be computed
        if not x.requires_grad: 
            return
        
        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # Reshape the gradient to match the original tensor shape
        grad_back = out_grad.reshape(x.data.shape)
        
        # Accumulate the gradient in the input tensor
        accumulate_gradient(x, grad_back)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)


def repeat(x: 'Tensor', repeats: int, axis: Optional[int] = None) -> 'Tensor':
    """
    Repeat elements of a tensor along a specified axis.
    
    Parameters:
    - x (Tensor): Input tensor.
    - repeats (int): Number of repetitions for each element.
    - axis (Optional[int]): The axis along which to repeat. If None, the array is flattened before repeating, and the output will be 1D.
    
    Returns:
    - Tensor: A new tensor with repeated elements.
    """
    
    # Define the forward function
    def forward() -> np.ndarray:
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
            
        # Return the repeated tensor
        return out_data
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the gradient needs to be computed
        if not x.requires_grad: 
            return
        
        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # If the axis is None, compute the gradient for the flattened tensor
        if axis is None:
            # Compute the gradient of the repeated tensor
            repeat_gradient(out_grad.ravel(), repeats, x.grad.ravel())
            
        # If the axis is specified, compute the gradient along that axis
        else:
            # Reduce the gradient along the specified axis
            grad_unrepeated = np.add.reduce(
                out_grad.reshape(
                    *(x.data.shape[:axis]), x.data.shape[axis], repeats, *x.data.shape[axis+1:]
                ), axis = axis+1
            )
            
            # Accumulate the gradient
            accumulate_gradient(x, grad_unrepeated)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)


def pad(x: 'Tensor', pad_width: Tuple[Tuple[int, int], ...]) -> 'Tensor':
    """
    Pad the tensor with zeros or a constant value.

    Parameters:
    - x (Tensor): Input tensor.
    - pad_width (Tuple[Tuple[int, int], ...]): Tuple of pad widths for each dimension. For example, for a 2D tensor, ((pad_top, pad_bottom), (pad_left, pad_right)).
    - mode (str): Padding mode (default "constant").

    Returns:
    - Tensor: A new tensor with padded data.
    """
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Extract the padding widths for each dimension and the input tensor shape
        (_, _), (pt, pb), (pl, pr), (_, _) = pad_width
        batch_size, height, width, channels = x.data.shape
        
        # Create the output tensor with the new shape
        out_data = np.empty((batch_size, height + pt + pb, width + pl + pr, channels), dtype=x.data.dtype)
        
        # Perform the padding operation
        pad_forward(x.data, pt, pb, pl, pr, out_data)
        
        # Return the padded tensor
        return out_data
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the gradient needs to be computed
        if not x.requires_grad: 
            return
        
        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # Extract the padding widths for each dimension
        (_, _), (pt, pb), (pl, pr), (_, _) = pad_width
        
        # Compute the gradient of the padding operation
        pad_gradient(out_grad, pt, pb, pl, pr, x.grad)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)


def conv_2d(x: 'Tensor', kernel: 'Tensor', stride: Tuple[int, int] = (1, 1)) -> 'Tensor':
    """
    Compute the 2D convolution of the input tensor with the kernel.
    
    Parameters:
    - x (Tensor): Input tensor.
    - kernel (Tensor): Convolution kernel.
    - stride (Tuple[int, int]): Stride of the convolution. Default is (1, 1).
    
    Returns:
    - Tensor: The result of the 2D convolution.
    
    Raises:
    - AssertionError: If the input channels do not match the kernel channels.
    - ValueError: If the kernel or stride is too large for the input size.
    """
    
    # Extract the stride values
    stride_height, stride_width = stride
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Extract the input dimensions
        batch_size, height, width, channels = x.data.shape
        out_channels, kernel_in_channels, kernel_height, kernel_width = kernel.data.shape
        
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
        
        # Return the output data
        return out_data
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Gradient for kernel
        if kernel.requires_grad:
            # Check if the gradient is initialized
            assert kernel.grad is not None, "Gradient must be initialized"
            
            # Compute the gradient with respect to the kernel
            conv_2d_gradient_w(x.data, out_grad, stride_height, stride_width, kernel.grad)
            
        # Gradient for input
        if x.requires_grad:
            # Check if the gradient is initialized
            assert x.grad is not None, "Gradient must be initialized"
            
            # Compute the gradient with respect to the input
            conv_2d_gradient_x(out_grad, kernel.data, stride_height, stride_width, x.grad)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op((x, kernel), forward, backward)


def max_pool_2d(x: 'Tensor', kernel_size: Tuple[int,int] = (2, 2), stride: Tuple[int,int] = (2, 2)) -> 'Tensor':
    """
    Apply a 2D Max Pooling over an input tensor of shape (N, H, W, C).
    
    Parameters:
    - x (Tensor): The input tensor, shape (N, H, W, C).
    - kernel_size (Tuple[int,int]): The spatial size of the window (kH, kW).
    - stride (Tuple[int,int]): The stride (stride_height, stride_width).
    
    Returns:
    - Tensor: The output after applying Max Pool 2D, shape (N, outH, outW, C).
    
    Raises:
    - ValueError: If the window or stride is too large for the input size.
    """
    
    # Define the stride values and initialize the indices for max pooling
    stride_height, stride_width = stride
    arg_i: np.ndarray
    arg_j: np.ndarray
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Set the nonlocal variables for max pooling indices
        nonlocal arg_i, arg_j
        
        # Extract the input dimensions
        batch_size, height, width, channels = x.data.shape
        kernel_height, kernel_width = kernel_size
        
        # Compute the output dimensions
        out_height = (height - kernel_height) // stride_height + 1
        out_width = (width - kernel_width) // stride_width + 1
        
        # Check if the kernel or stride is too large for the input size
        if out_height < 1 or out_width < 1:
            raise ValueError("Kernel size or stride too large for input size.")

        # Create the output array
        out_data = np.empty((batch_size, out_height, out_width, channels), dtype=x.data.dtype)
        
        # Initialize the indices for max pooling
        arg_i = np.zeros_like(out_data, dtype=np.int32)
        arg_j = np.zeros_like(out_data, dtype=np.int32)

        # Perform the max pooling operation
        max_pool_2d_forward(x.data, kernel_height, kernel_width, stride_height, stride_width, out_data, arg_i, arg_j)
        
        # Return the output data
        return out_data
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the gradient needs to be computed
        if not x.requires_grad: 
            return
        
        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # Backprop the gradient through the max pooling operation
        max_pool_2d_gradient(arg_i, arg_j, out_grad, stride_height, stride_width, x.grad)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)