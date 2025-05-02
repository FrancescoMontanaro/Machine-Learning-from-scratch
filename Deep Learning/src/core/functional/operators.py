import numpy as np
from functools import partial
from typing import Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING: from ..tensor import Tensor
from .base import tensor_unary_op, tensor_binary_op, accumulate_gradient

# Import the necessary kernel functions
from .kernel.add import add_forward, add_backward
from .kernel.sub import sub_forward, sub_backward_a, sub_backward_b
from .kernel.mul import mul_forward, mul_backward_a, mul_backward_b
from .kernel.div import div_forward, div_backward_a, div_backward_b
from .kernel.pow import pow_forward, pow_gradient


def add(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to add two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Sum of the two tensors
    """
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op(
        input = (a, b), 
        forward_fn = add_forward, 
        backward_fn_a = partial(add_backward, target_shape=a.data.shape),
        backward_fn_b = partial(add_backward, target_shape=b.data.shape)
    )


def sub(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to subtract two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Difference of the two tensors
    """
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op(
        input = (a, b), 
        forward_fn = sub_forward,
        backward_fn_a = partial(sub_backward_a, target_shape=a.data.shape),
        backward_fn_b = partial(sub_backward_b, target_shape=b.data.shape)
    )


def mul(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to multiply two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Product of the two tensors
    """
            
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op(
        input = (a, b),
        forward_fn = mul_forward,
        backward_fn_a = partial(mul_backward_a, target_shape=a.data.shape, b_data=b.data),
        backward_fn_b = partial(mul_backward_b, target_shape=b.data.shape, a_data=a.data)
    )


def div(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to divide two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Quotient of the two tensors
    """
            
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op(
        input = (a, b),
        forward_fn = div_forward, 
        backward_fn_a = partial(div_backward_a, target_shape=a.data.shape, b_data=b.data),
        backward_fn_b = partial(div_backward_b, a_data=a.data, b_data=b.data)
    )


def mat_mul(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to perform matrix multiplication between two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Matrix product of the two tensors
    """
    
    # Define the forward function
    def forward(a_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
        # Compute the matrix product of the two tensors
        return np.matmul(a_data, b_data)
    
    # Define the backward function
    def backward_a(out_grad: np.ndarray, b_data: np.ndarray, out_buffer: np.ndarray) -> None:
        # Compute the gradients for the inputs of the matrix multiplication operation
        out_buffer += np.matmul(out_grad, np.swapaxes(b_data, -1, -2))
    
    def backward_b(out_grad: np.ndarray, a_data: np.ndarray, out_buffer: np.ndarray) -> None:
        # Compute the gradient for the second input tensor
        out_buffer += np.matmul(np.swapaxes(a_data, -1, -2), out_grad)
            
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op(
        input = (a, b),
        forward_fn = forward,
        backward_fn_a = partial(backward_a, b_data=b.data),
        backward_fn_b = partial(backward_b, a_data=a.data)
    )


def pow(x: 'Tensor', power: Union[int, float]) -> 'Tensor':
    """
    Function to raise a tensor to the power of another tensor.
    
    Parameters:
    - x (Tensor): Base tensor
    - power (Union[int, float]): Exponent tensor
    
    Returns:
    - Tensor: Base tensor raised to the power of the exponent tensor
    
    Raises:
    - AssertionError: If the power is not an integer or a float
    """
    
    # Ensure the power is a scalar
    assert isinstance(power, (int, float)), "The power must be an integer or a float"
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Compute the power of the tensor
        return pow_forward(x.data, power)
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the tensor requires gradient computation
        if not x.requires_grad:
            return
        
        # Compute the gradient of the power function
        grad = pow_gradient(out_grad, x.data, power)
        
        # Accumulate the gradient into self.grad.
        accumulate_gradient(x, grad)
        
    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)


def get_item(x: 'Tensor', key: Union[int, slice, np.ndarray, Tuple[Union[int, slice, np.ndarray], ...]]) -> 'Tensor':
    """
    Function to slice a tensor.
    
    Parameters:
    - x (Tensor): Input tensor
    - key (Union[int, slice, np.ndarray, Tuple[Union[int, slice, np.ndarray], ...]]): Index or slice to extract
    
    Returns:
    - Tensor: Sliced tensor
    """
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Return the sliced data
        return x.data[key]
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the tensor requires gradient computation
        if not x.requires_grad:
            return
        
        # Check if the gradient is initialized
        assert x.grad is not None, "Gradient must be initialized"
        
        # Compute the gradient of the slice operation
        grad = np.zeros_like(x.data)
        
        # Use direct assignment which handles slices correctly
        np.add.at(grad, key, out_grad) # type: ignore
                
        # Accumulate the gradient into self.grad.
        accumulate_gradient(x, grad)
        
    # Return the tensor operation with the specified forward and backward functions
    return tensor_unary_op(x, forward, backward)