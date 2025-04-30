import numpy as np
from typing import Callable, Union, Type, List, Tuple, TYPE_CHECKING, cast

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


def tensor_op(input: Union['Tensor', List['Tensor'], Tuple['Tensor', ...]], forward_fn: Callable[..., np.ndarray], backward_fn: Callable[..., None]) -> 'Tensor':
    """
    Function to create a tensor operation with automatic differentiation.
    
    Parameters:
    - input (Union[Tensor, List[Tensor], Tuple[Tensor]]): Input tensor(s) for the operation.
    - forward_fn (Callable[..., np.ndarray]): Forward function that computes the output.
    - backward_fn (Callable[..., None]): Backward function that computes the gradients.
    
    Returns:
    - Tensor: Output tensor with the computed gradients.
    
    Raises:
    - TypeError: If any input is not a Tensor.
    """
    
    # Get the tensor class
    TensorCls = cast(Type['Tensor'], get_tensor_class())
    
    # The input is a list of tensors
    if isinstance(input, list):
        # Check if the inputs are tensors
        if not all(isinstance(t, TensorCls) for t in input):
            raise TypeError("All inputs must be instances of Tensor.")
        
        # Extract the raw arrays
        data = [t.data for t in input]
        
        # Call the forward function
        out_data = forward_fn(data)
        
        # Check if any input tensor requires gradients
        requires_grad = any(t.requires_grad for t in input)
        
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
            for t in input:
                # If the tensor requires gradients, set its gradient to zero
                if t.requires_grad and t.grad is None:
                    # Initialize the gradient to zero
                    t.grad = np.zeros_like(t.data, dtype=t.data.dtype)
                    
            # Call the backward function with the output gradient
            backward_fn(input, out.grad)
            
        # Register the input tensors that require gradients in the computation graph
        prev = {t for t in input if t.requires_grad}
        
    # The input is a tuple of tensors
    elif isinstance(input, tuple):
        # Check if the inputs are tensors
        if not all(isinstance(t, TensorCls) for t in input):
            raise TypeError("All inputs must be instances of Tensor.")
        
        # Extract the raw arrays
        data = tuple(t.data for t in input)
        
        # Call the forward function
        out_data = forward_fn(data)
        
        # Check if any input tensor requires gradients
        requires_grad = any(t.requires_grad for t in input)
        
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
            for t in input:
                # If the tensor requires gradients, set its gradient to zero
                if t.requires_grad and t.grad is None:
                    # Initialize the gradient to zero
                    t.grad = np.zeros_like(t.data, dtype=t.data.dtype)
                    
            # Call the backward function with the output gradient
            backward_fn(input, out.grad)
            
        # Register the input tensors that require gradients in the computation graph
        prev = {t for t in input if t.requires_grad}
       
    # The input is a single tensor     
    else:
        # Check if the input is a tensor
        if not isinstance(input, TensorCls):
            raise TypeError("Input must be an instance of Tensor.")
        
        # Extract the raw array
        data = input.data
        
        # Call the forward function
        out_data = forward_fn(data)
        
        # Create output tensor
        out = TensorCls(out_data, requires_grad=input.requires_grad)
        
        # If no gradients are required, return the output tensor without backward
        if _NO_GRAD: return out
        
        # Define the backward function
        def _backward():
            # If the output tensor is None, return
            if out.grad is None:
                return
            
            # If the tensor requires gradients, set its gradient to zero
            if input.requires_grad and input.grad is None:
                # Initialize the gradient to zero
                input.grad = np.zeros_like(input.data, dtype=input.data.dtype)
                
            # Call the backward function with the output gradient
            backward_fn(input, out.grad)
            
        # Register the input tensor that requires gradients in the computation graph
        prev = {input} if input.requires_grad else set()
        
    # Set the backward function to the output tensor
    out._backward = _backward
    
    # Set the previous tensors that require gradients
    out._prev = prev
    
    # Return the output tensor
    return out