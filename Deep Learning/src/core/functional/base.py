import numpy as np
from typing import Callable, List, TYPE_CHECKING

from .tape import tape_pop
from .kernel import add_forward
if TYPE_CHECKING: from ..tensor import Tensor
from ..utils.context_manager import _NO_GRAD


def accumulate_gradient(x: 'Tensor', grad: np.ndarray) -> None:
    """
    Accumulates gradients for a tensor.

    Parameters:
    - x (Tensor): Input tensor.
    - grad (Tensor): Gradient tensor.
    """
    
    # Accumulate the gradient
    x.grad = add_forward(x.grad, grad.astype(x.grad.dtype)) if x.grad is not None else grad


def tensor_unary_op(t: 'Tensor', forward_fn: Callable[..., tuple[np.ndarray, int]], backward_fn: Callable[..., None], tensor_cls: type['Tensor']) -> 'Tensor':
    """
    Function to create a tensor operation with automatic differentiation.
    
    Parameters:
    - t (Tensor): Input tensor for the operation.
    - forward_fn (Callable[..., np.ndarray]): Forward function that computes the output.
    - backward_fn (Callable[..., None]): Backward function that computes the gradients.
    - tensor_cls (type[Tensor]): The Tensor class to be used for creating the output tensor.
    
    Returns:
    - Tensor: Output tensor with the computed gradients.
    """
    
    # Call the forward function
    out_data, tape_idx = forward_fn(t.data)
    
    # Check if the gradient is required
    if _NO_GRAD or not t.requires_grad:
        # If no gradients are required, return the output tensor without backward
        return tensor_cls(out_data, requires_grad=False)
    
    # Create output tensor
    out = tensor_cls(out_data, requires_grad=True)
    
    # Define the backward function
    def _backward():
        # If the output tensor is None, return
        if out.grad is None:
            return
        
        # Retrieve the saved data from the tape
        saved_data = tape_pop(tape_idx)
        
        # If the tensor requires gradients, set its gradient to zero
        if t.grad is None:
            # Initialize the gradient to zero
            t.grad = np.zeros_like(t.data, dtype=t.data.dtype)
            
        # Call the backward function with the output gradient
        backward_fn(out_grad=out.grad, out_buffer=t.grad, saved_data=saved_data)
        
    # Set the backward function to the output tensor
    out._backward = _backward
    
    # Set the previous tensors
    if t.requires_grad: out._prev = {t}
    
    # Return the output tensor
    return out


def tensor_binary_op(t1: 'Tensor', t2: 'Tensor', forward_fn: Callable[..., tuple[np.ndarray, int]], backward_fn_a: Callable[..., None], backward_fn_b: Callable[..., None], tensor_cls: type['Tensor']) -> 'Tensor':
    """
    Function to create a tensor operation with automatic differentiation.
    
    Parameters:
    - t1 (Tensor): First input tensor for the operation.
    - t2 (Tensor): Second input tensor for the operation.
    - forward_fn (Callable[..., np.ndarray]): Forward function that computes the output.
    - backward_fn_a (Callable[..., None]): Backward function for the first input tensor.
    - backward_fn_b (Callable[..., None]): Backward function for the second input tensor.
    - tensor_cls (type[Tensor]): The Tensor class to be used for creating the output tensor.
    
    Returns:
    - Tensor: Output tensor with the computed gradients.
    """
    
    # Call the forward function
    out_data, tape_idx = forward_fn(t1.data, t2.data)
    
    # Check if the gradient is required
    if _NO_GRAD or not (t1.requires_grad or t2.requires_grad):
        # If no gradients are required, return the output tensor without backward
        return tensor_cls(out_data, requires_grad=False)
    
    # Create output tensor
    out = tensor_cls(out_data, requires_grad=True)
    
    # Define the backward function
    def _backward():
        # If the output tensor is None, return
        if out.grad is None:
            return
        
        # Retrieve the saved data from the tape
        saved_data = tape_pop(tape_idx)
        
        # Check if the first tensor requires gradients
        if t1.requires_grad:
            # If the gradient is None, initialize it to zero
            if t1.grad is None:
                t1.grad = np.zeros_like(t1.data, dtype=t1.data.dtype)
                
            # Backward pass for the first tensor
            backward_fn_a(out.grad, t1.grad, saved_data)
            
        # Check if the second tensor requires gradients
        if t2.requires_grad:
            # If the gradient is None, initialize it to zero
            if t2.grad is None:
                t2.grad = np.zeros_like(t2.data, dtype=t2.data.dtype)
                
            # Backward pass for the second tensor
            backward_fn_b(out.grad, t2.grad, saved_data)
        
    # Set the backward function to the output tensor
    out._backward = _backward
    
    # Set the previous tensors
    out._prev = {t for t in (t1, t2) if t.requires_grad}
    
    # Return the output tensor
    return out


def tensor_nary_op(tensors: List['Tensor'], forward_fn: Callable[..., tuple[np.ndarray, int]], backward_fn: Callable[..., None], tensor_cls: type['Tensor']) -> 'Tensor':
    """
    Function to create a tensor operation with automatic differentiation.
    
    Parameters:
    - tensors (List[Tensor]): Input tensors for the operation.
    - forward_fn (Callable[..., np.ndarray]): Forward function that computes the output.
    - backward_fn (Callable[..., None]): Backward function that computes the gradients.
    - tensor_cls (type[Tensor]): The Tensor class to be used for creating the output tensor.
    
    Returns:
    - Tensor: Output tensor with the computed gradients.
    """
    
    # Call the forward function
    out_data, tape_idx = forward_fn([t.data for t in tensors])
    
    # Check if the gradient is required
    if _NO_GRAD or not any(t.requires_grad for t in tensors):
        # If no gradients are required, return the output tensor without backward
        return tensor_cls(out_data, requires_grad=False)
    
    # Create output tensor
    out = tensor_cls(out_data, requires_grad=True)
    
    # Define the backward function
    def _backward():
        # If the output tensor is None, return
        if out.grad is None:
            return
        
        # Retrieve the saved data from the tape
        saved_data = tape_pop(tape_idx)
        
        # Iterate over the input tensors
        for idx, t in enumerate(tensors):
            # Check if the tensor requires gradients
            if t.requires_grad:
                # If the gradient is None, initialize it to zero
                if t.grad is None:
                    # Initialize the gradient to zero
                    t.grad = np.zeros_like(t.data, dtype=t.data.dtype)
                
                # Call the backward function with the output gradient
                backward_fn(out_grad=out.grad, out_buffer=t.grad, tensor_idx=idx, saved_data=saved_data)
        
    # Set the backward function to the output tensor
    out._backward = _backward
    
    # Set the previous tensors
    out._prev = {t for t in tensors if t.requires_grad}
    
    # Return the output tensor
    return out