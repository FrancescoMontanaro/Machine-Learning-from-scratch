import numpy as np
from typing import Type, TYPE_CHECKING, cast

from .utils import accumulate_gradient
if TYPE_CHECKING: from ..tensor import Tensor
from ..utils.context_manager import _NO_GRAD
from ..utils.types_registry import get_tensor_class


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
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Compute the derivative of the sigmoid function
            grad = out.data * (1 - out.data) * out.grad
            
            # Update the gradient of the input tensor
            accumulate_gradient(x, grad)
            
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
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Compute the gradient
            grad = out.grad * (x.data > 0)
            
            # Accumulate the gradient
            accumulate_gradient(x, grad)
            
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
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Compute the derivative of the tanh function
            grad = out.grad * (1 - np.tanh(x.data) ** 2)
            
            # Accumulate the gradient
            accumulate_gradient(x, grad)
            
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
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Weighted sum of the gradients along the specified axis
            g_sum = np.sum(out.grad * out.data, axis=axis, keepdims=True)
            
            # Compute the gradient of the loss with respect to the current tensor
            grad_input = out.data * (out.grad - g_sum)
            
            # Add the gradient to the current tensor
            accumulate_gradient(x, grad_input)

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
    - axis (int): Axis along which to compute softmax. Default: -1 (last axis)

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
    
    # If gradient computation is disabled, return the output tensor without a backward function
    if _NO_GRAD: return out
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Compute softmax (exp(log_softmax))
            softmax = np.exp(out.data)
            
            # Sum of gradients along the softmax axis
            g_sum = np.sum(out.grad, axis=axis, keepdims=True)
            
            # Compute gradient
            grad_input = out.grad - softmax * g_sum
            
            # Accumulate gradient
            accumulate_gradient(x, grad_input)

    # Store the backward function
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    return out