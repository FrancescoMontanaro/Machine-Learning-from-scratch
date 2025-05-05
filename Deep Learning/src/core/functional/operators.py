import numpy as np
from typing import Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING: from ..tensor import Tensor
from .base import Context, tensor_unary_op, tensor_binary_op, accumulate_gradient

# Import the necessary kernel functions
from .kernel.add import add_forward, add_backward
from .kernel.pow import pow_forward, pow_gradient
from .kernel.sub import sub_forward, sub_backward_a, sub_backward_b
from .kernel.mul import mul_forward, mul_backward_a, mul_backward_b
from .kernel.div import div_forward, div_backward_a, div_backward_b
from .kernel.mat_mul import matmul_forward, matmul_backward_a, matmul_backward_b


def add(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to add two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Sum of the two tensors
    """
    
    # Define the forward function
    def forward(ctx: Context, a_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
        # Save the data for backward pass
        ctx.save(a_data=a_data, b_data=b_data)
        
        # Compute the sum of the two tensors
        return add_forward(a_data, b_data)
    
    # Define the backward function for tensor a
    def backward_a(ctx: Context, out_grad: np.ndarray, out_buffer: np.ndarray) -> None:
        # Compute the gradient of the sum operation
        add_backward(out_grad=out_grad, out_buffer=out_buffer, target_shape=ctx.a_data.shape)
        
    # Define the backward function for tensor b
    def backward_b(ctx: Context, out_grad: np.ndarray, out_buffer: np.ndarray) -> None:
        # Compute the gradient of the sum operation
        add_backward(out_grad=out_grad, out_buffer=out_buffer, target_shape=ctx.b_data.shape)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op(
        t1 = a,
        t2 = b, 
        forward_fn = forward, 
        backward_fn_a = backward_a,
        backward_fn_b = backward_b
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
    
    # Define the forward function
    def forward(ctx: Context, a_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
        # Save the data for backward pass
        ctx.save(a_data=a_data, b_data=b_data)
        
        # Compute the difference of the two tensors
        return sub_forward(a_data, b_data)
    
    # Define the backward function for tensor a
    def backward_a(ctx: Context, out_grad: np.ndarray, out_buffer: np.ndarray) -> None:
        # Compute the gradient of the subtraction operation
        sub_backward_a(out_grad=out_grad, out_buffer=out_buffer, target_shape=ctx.a_data.shape)
        
    # Define the backward function for tensor b
    def backward_b(ctx: Context, out_grad: np.ndarray, out_buffer: np.ndarray) -> None:
        # Compute the gradient of the subtraction operation
        sub_backward_b(out_grad=out_grad, out_buffer=out_buffer, target_shape=ctx.b_data.shape)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op(
        t1 = a,
        t2 = b,
        forward_fn = forward,
        backward_fn_a = backward_a,
        backward_fn_b = backward_b
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
    
    # Define the forward function
    def forward(ctx: Context, a_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
        # Save the data for backward pass
        ctx.save(a_data=a_data, b_data=b_data)
        
        # Compute the product of the two tensors
        return mul_forward(a_data, b_data)
    
    # Define the backward function for tensor a
    def backward_a(ctx: Context, out_grad: np.ndarray, out_buffer: np.ndarray) -> None:
        # Compute the gradient of the multiplication operation
        mul_backward_a(out_grad=out_grad, out_buffer=out_buffer, target_shape=ctx.a_data.shape, b_data=ctx.b_data)
        
    # Define the backward function for tensor b
    def backward_b(ctx: Context, out_grad: np.ndarray, out_buffer: np.ndarray) -> None:
        # Compute the gradient of the multiplication operation
        mul_backward_b(out_grad=out_grad, out_buffer=out_buffer, target_shape=ctx.b_data.shape, a_data=ctx.a_data)
            
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op(
        t1 = a,
        t2 = b,
        forward_fn = forward,
        backward_fn_a = backward_a,
        backward_fn_b = backward_b
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
    
    # Define the forward function
    def forward(ctx: Context, a_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
        # Save the data for backward pass
        ctx.save(a_data=a_data, b_data=b_data)
        
        # Compute the quotient of the two tensors
        return div_forward(a_data, b_data)
    
    # Define the backward function for tensor a
    def backward_a(ctx: Context, out_grad: np.ndarray, out_buffer: np.ndarray) -> None:
        # Compute the gradient of the division operation
        div_backward_a(out_grad=out_grad, out_buffer=out_buffer, target_shape=ctx.a_data.shape, b_data=ctx.b_data)
        
    # Define the backward function for tensor b
    def backward_b(ctx: Context, out_grad: np.ndarray, out_buffer: np.ndarray) -> None:
        # Compute the gradient of the division operation
        div_backward_b(out_grad=out_grad, out_buffer=out_buffer, a_data=ctx.a_data, b_data=ctx.b_data)
            
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op(
        t1 = a,
        t2 = b,
        forward_fn = forward, 
        backward_fn_a = backward_a,
        backward_fn_b = backward_b
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
    def forward(ctx: Context, a_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
        # Save the data for backward pass
        ctx.save(a_data=a_data, b_data=b_data)
        
        # Compute the matrix product of the two tensors
        return matmul_forward(a_data, b_data)
    
    # Define the backward function for tensor a
    def backward_a(ctx: Context, out_grad: np.ndarray, out_buffer: np.ndarray) -> None:
        # Compute the gradient of the matrix multiplication operation
        matmul_backward_a(out_grad=out_grad, out_buffer=out_buffer, b_data=ctx.b_data)
        
    # Define the backward function for tensor b
    def backward_b(ctx: Context, out_grad: np.ndarray, out_buffer: np.ndarray) -> None:
        # Compute the gradient of the matrix multiplication operation
        matmul_backward_b(out_grad=out_grad, out_buffer=out_buffer, a_data=ctx.a_data)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op(
        t1 = a,
        t2 = b,
        forward_fn = forward,
        backward_fn_a = backward_a,
        backward_fn_b = backward_b
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