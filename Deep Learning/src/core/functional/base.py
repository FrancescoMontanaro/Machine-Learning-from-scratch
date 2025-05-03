import numpy as np
from typing import Callable, Type, List, TYPE_CHECKING, cast

if TYPE_CHECKING: from ..tensor import Tensor
from ..utils.context_manager import _NO_GRAD
from ..utils.types_registry import get_tensor_class


def accumulate_gradient(x: 'Tensor', grad: np.ndarray) -> None:
    """
    Accumulates gradients for a tensor.

    Parameters:
    - x (Tensor): Input tensor.
    - grad (Tensor): Gradient tensor.
    """
    
    # Accumulate the gradient
    x.grad = x.grad + grad if x.grad is not None else grad


def tensor_unary_op(t: 'Tensor', forward_fn: Callable[..., np.ndarray], backward_fn: Callable[..., None]) -> 'Tensor':
    """
    Function to create a tensor operation with automatic differentiation.
    
    Parameters:
    - t (Tensor): Input tensor for the operation.
    - forward_fn (Callable[..., np.ndarray]): Forward function that computes the output.
    - backward_fn (Callable[..., None]): Backward function that computes the gradients.
    
    Returns:
    - Tensor: Output tensor with the computed gradients.
    
    Raises:
    - TypeError: If any input is not a Tensor.
    """
    
    # Get the tensor class
    TensorCls = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    if not isinstance(t, TensorCls):
        raise TypeError("Input must be an instance of Tensor.")
    
    # Call the forward function
    out_data = forward_fn()
    
    # Create output tensor
    out = TensorCls(out_data, requires_grad=t.requires_grad)
    
    # If no gradients are required, return the output tensor without backward
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward():
        # If the output tensor is None, return
        if out.grad is None:
            return
        
        # If the tensor requires gradients, set its gradient to zero
        if t.requires_grad and t.grad is None:
            # Initialize the gradient to zero
            t.grad = np.zeros_like(t.data, dtype=t.data.dtype)
            
        # Call the backward function with the output gradient
        backward_fn(out.grad)
        
    # Set the backward function to the output tensor
    out._backward = _backward
    
    # Set the previous tensors that require gradients
    out._prev = {t} if t.requires_grad else set()
    
    # Return the output tensor
    return out


def tensor_binary_op(t1: 'Tensor', t2: 'Tensor', forward_fn: Callable[..., np.ndarray], backward_fn_a: Callable[..., None], backward_fn_b: Callable[..., None]) -> 'Tensor':
    """
    Function to create a tensor operation with automatic differentiation.
    
    Parameters:
    - t1 (Tensor): First input tensor for the operation.
    - t2 (Tensor): Second input tensor for the operation.
    - forward_fn (Callable[..., np.ndarray]): Forward function that computes the output.
    - backward_fn_a (Callable[..., None]): Backward function for the first input tensor.
    - backward_fn_b (Callable[..., None]): Backward function for the second input tensor.
    
    Returns:
    - Tensor: Output tensor with the computed gradients.
    
    Raises:
    - TypeError: If any input is not a Tensor.
    """
    
    # Get the tensor class
    TensorCls = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the inputs are tensors
    if not isinstance(t1, TensorCls) or not isinstance(t2, TensorCls):
        raise TypeError("All inputs must be instances of Tensor.")
    
    # Call the forward function
    out_data = forward_fn(t1.data, t2.data)
    
    # Check if any input tensor requires gradients
    requires_grad = t1.requires_grad or t2.requires_grad
    
    # Create output tensor
    out = TensorCls(out_data, requires_grad=requires_grad)
    
    # If no gradients are required, return the output tensor without backward
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward():
        # If the output tensor is None, return
        if out.grad is None:
            return
        
        # Check if the first tensor requires gradients
        if t1.requires_grad:
            # If the gradient is None, initialize it to zero
            if t1.grad is None:
                t1.grad = np.zeros_like(t1.data, dtype=t1.data.dtype)
                
            # Backward pass for the first tensor
            backward_fn_a(out_grad=out.grad, out_buffer=t1.grad)
            
        # Check if the second tensor requires gradients
        if t2.requires_grad:
            # If the gradient is None, initialize it to zero
            if t2.grad is None:
                t2.grad = np.zeros_like(t2.data, dtype=t2.data.dtype)
                
            # Backward pass for the second tensor
            backward_fn_b(out_grad=out.grad, out_buffer=t2.grad)
        
    # Set the backward function to the output tensor
    out._backward = _backward
    
    # Set the previous tensors that require gradients
    out._prev = {t for t in (t1, t2) if t.requires_grad}
    
    # Return the output tensor
    return out


def tensor_nary_op(tensors: List['Tensor'], forward_fn: Callable[..., np.ndarray], backward_fn: Callable[..., None]) -> 'Tensor':
    """
    Function to create a tensor operation with automatic differentiation.
    
    Parameters:
    - tensors (List[Tensor]): Input tensors for the operation.
    - forward_fn (Callable[..., np.ndarray]): Forward function that computes the output.
    - backward_fn (Callable[..., None]): Backward function that computes the gradients.
    
    Returns:
    - Tensor: Output tensor with the computed gradients.
    
    Raises:
    - TypeError: If any input is not a Tensor.
    """
    
    # Get the tensor class
    TensorCls = cast(Type['Tensor'], get_tensor_class())

    # Check if the inputs are tensors
    if not all(isinstance(t, TensorCls) for t in tensors):
        raise TypeError("All inputs must be instances of Tensor.")
    
    # Call the forward function
    out_data = forward_fn()
    
    # Check if any input tensor requires gradients
    requires_grad = any(t.requires_grad for t in tensors)
    
    # Create output tensor
    out = TensorCls(out_data, requires_grad=requires_grad)
    
    # If no gradients are required, return the output tensor without backward
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward():
        # If the output tensor is None, return
        if out.grad is None:
            return
        
        # Iterate over the input tensors
        for t in tensors:
            # If the tensor requires gradients, set its gradient to zero
            if t.requires_grad and t.grad is None:
                # Initialize the gradient to zero
                t.grad = np.zeros_like(t.data, dtype=t.data.dtype)
                
        # Call the backward function with the output gradient
        backward_fn(out.grad)
        
    # Set the backward function to the output tensor
    out._backward = _backward
    
    # Set the previous tensors that require gradients
    out._prev = {t for t in tensors if t.requires_grad}
    
    # Return the output tensor
    return out