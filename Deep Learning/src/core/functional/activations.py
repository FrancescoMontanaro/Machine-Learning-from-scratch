import numpy as np
from typing import Type, TYPE_CHECKING, cast

from ..utils.registry import get_tensor_class
if TYPE_CHECKING: from ..tensor import Tensor


def sigmoid(x: 'Tensor') -> 'Tensor':
    """
    Compute the sigmoid activation function.

    Parameters:
    - x (Tensor): Input tensor

    Returns:
    - Tensor: Output tensor
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Prepare an empty array for the output
    out_data = np.empty_like(x.data)
    
    # Create a boolean mask for non-negative values
    mask = x.data >= 0
    
    # For x >= 0: compute 1 / (1 + exp(-x))
    out_data[mask] = 1 / (1 + np.exp(-x.data[mask]))
    
    # For x < 0: compute exp(x) / (1 + exp(x))
    # Only compute exp for the negative part to avoid overflow
    exp_x = np.exp(x.data[~mask])
    out_data[~mask] = exp_x / (1 + exp_x)
    
    # Compute the sigmoid of the tensor
    out = Tensor(data=out_data, requires_grad=x.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Compute the derivative of the sigmoid function
            grad = out.data * (1 - out.data) * out.grad
            
            # Update the gradient of the input tensor
            x.grad = x.grad + grad if x.grad is not None else grad
            
    # Store the backward function with respect to the sigmoid operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    # Return the output tensor
    return out


def relu(x: 'Tensor') -> 'Tensor':
    """
    Compute the ReLU activation function.

    Parameters:
    - x (Tensor): Input tensor

    Returns:
    - Tensor: Output tensor
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Compute the ReLU of the tensor
    out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Compute the gradient of the loss with respect to the current tensor
            x.grad = x.grad + (x.data > 0) * out.grad if x.grad is not None else (x.data > 0) * out.grad
            
    # Store the backward function with respect to the ReLU operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    # Return the output tensor
    return out


def tanh(x: 'Tensor') -> 'Tensor':
    """
    Compute the tanh activation function.

    Parameters:
    - x (Tensor): Input tensor

    Returns:
    - Tensor: Output tensor
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Compute the hyperbolic tangent of the tensor
    out = Tensor(np.tanh(x.data), requires_grad=x.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:                
            # Compute the gradient of the loss with respect to the current tensor
            x.grad = x.grad + (1 - np.tanh(x.data) ** 2) * out.grad if x.grad is not None else (1 - np.tanh(x.data) ** 2) * out.grad
            
    # Store the backward function with respect to the hyperbolic tangent operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    # Return the output tensor
    return out


def softmax(x: 'Tensor', axis: int = -1) -> 'Tensor':
    """
    Compute the softmax activation function.

    Parameters:
    - x (Tensor): Input tensor

    Returns:
    - Tensor: Output tensor
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Shift the input tensor to avoid numerical instability
    x_shifted = x - x.max(axis=axis, keepdims=True)
    
    # Compute exponentials and sum them along the specified axis
    exp_x = x_shifted.exp()
    sum_exp = exp_x.sum(axis=axis, keepdims=True)
    
    # Crea un nuovo Tensor con i dati di s
    out = exp_x / sum_exp
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Weighted sum of the gradients along the specified axis
            g_sum = np.sum(out.grad * out.data, axis=axis, keepdims=True)
            
            # Compute the gradient of the loss with respect to the current tensor
            grad_input = out.data * (out.grad - g_sum)
            
            # Add the gradient to the current tensor
            x.grad = x.grad + grad_input if x.grad is not None else grad_input

    # Store the backward function with respect to the softmax operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    # Return the output tensor
    return out


def log_softmax(x: 'Tensor', axis: int = -1) -> 'Tensor':
    """
    Compute the log softmax activation function.

    Parameters:
    - x (Tensor): Input tensor

    Returns:
    - Tensor: Output tensor
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Check if the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Shift the input tensor to avoid numerical instability
    x_shifted = x - x.max(axis=axis, keepdims=True)
    
    # Compute exponentials and sum them along the specified axis
    exp_x = x_shifted.exp()
    sum_exp = exp_x.sum(axis=axis, keepdims=True)
    
    # Compute the log softmax
    out = x_shifted - sum_exp.log()
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Weighted sum of the gradients along the specified axis
            g_sum = np.sum(out.grad, axis=axis, keepdims=True)
            
            # Compute the gradient of the loss with respect to the current tensor
            grad_input = out.grad - np.exp(out.data) * g_sum
            
            # Add the gradient to the current tensor
            x.grad = x.grad + grad_input if x.grad is not None else grad_input

    # Store the backward function with respect to the log softmax operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    # Return the output tensor
    return out