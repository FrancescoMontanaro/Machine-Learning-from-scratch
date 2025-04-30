import numpy as np
from typing import Type, TYPE_CHECKING, cast

if TYPE_CHECKING: from ..tensor import Tensor
from ..utils.context_manager import _NO_GRAD
from ..utils.types_registry import get_tensor_class

# Importing the kernel functions
from .kernel.relu import relu_forward, relu_gradient
from .kernel.tanh import tanh_forward, tanh_gradient
from .kernel.softmax import softmax_forward, softmax_gradient
from .kernel.sigmoid import sigmoid_forward, sigmoid_gradient
from .kernel.log_softmax import log_softmax_forward, log_softmax_gradient


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
    
    # Extract the number of elements in the input tensor
    n = x.data.size
    
    # Create an empty array to store the output data
    out_data = np.empty_like(x.data)
    
    # Compute the sigmoid function
    sigmoid_forward(x.data.ravel(), out_data.ravel(), n)
    
    # Create a new tensor with the output data
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
            
            # Compute the gradient of the sigmoid function
            sigmoid_gradient(out_data.ravel(), out.grad.ravel(), x.grad.ravel(), n)
            
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

    # Extract the number of elements in the input tensor
    n = x.data.size
    
    # Create an empty array to store the output data
    out_data = np.empty_like(x.data)
    
    # Compute the ReLU function
    relu_forward(x.data.ravel(), out_data.ravel(), n)
    
    # Create a new tensor with the output data
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
            
            # Compute the gradient of the ReLU function
            relu_gradient(x.data.ravel(), out.grad.ravel(), x.grad.ravel(), n)
            
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
    
    # Extract the number of elements in the input tensor
    n = x.data.size
    
    # Create an empty array to store the output data
    out_data = np.empty_like(x.data)
    
    # Compute the tanh function
    tanh_forward(x.data.ravel(), out_data.ravel(), n)
    
    # Create a new tensor with the output data
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
                
            tanh_gradient(out_data.ravel(), out.grad.ravel(), x.grad.ravel(), n)
            
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
    
    # Extract number of elements in the input tensor
    ndim = x.data.ndim
    
    # Compute the axis for softmax
    ax = axis % ndim
    
    # Compute the number of classes and the number of samples
    k = x.data.shape[-1]
    n = x.data.size // k
    
    # If the axis is not the last one, compute softmax along the specified axis
    if ax != ndim - 1:
        # Compute the maximum value along the specified axis
        out_data = np.exp(x.data) / np.sum(np.exp(x.data), axis=ax, keepdims=True)
    # If the axis is the last one, compute softmax using the kernel function
    else:
        # Create an empty array to store the output data
        out_data = np.empty_like(x.data)
        
        # Compute the softmax function
        softmax_forward(x.data.ravel(), out_data.ravel(), n, k)
        
    # Create a new tensor with the output data
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
            
            # Compute the gradient of the softmax function
            softmax_gradient(out_data.ravel(), out.grad.ravel(), x.grad.ravel(), n, k)

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
    
    # Extract the number of elements in the input tensor
    ndim = x.data.ndim
    
    # Compute the axis for log softmax
    ax = axis % ndim
    
    # Extract the number of classes and the number of samples
    k = x.data.shape[-1]
    n = x.data.size // k
    
    # If the axis is not the last one, compute log softmax along the specified axis
    if ax != ndim - 1:
        # Compute the maximum value along the specified axis
        m = np.max(x.data, axis=axis, keepdims=True)
        
        # Subtract the maximum value from the input data
        y = x.data - m
        
        # Compute the log sum of exponentials
        logsum = np.log(np.sum(np.exp(y), axis=axis, keepdims=True))
        
        # Compute the log softmax
        out_data = y - logsum
    
    # If the axis is the last one, compute log softmax using the kernel function
    else:
        # Create an empty array to store the output data
        out_data = np.empty_like(x.data)
        
        # Compute the log softmax function
        log_softmax_forward(x.data.ravel(), out_data.ravel(), n, k)
        
    # Create a new tensor with the output data
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
            
            # Compute the gradient of the log softmax function
            log_softmax_gradient(out_data.ravel(), out.grad.ravel(), x.grad.ravel(), n, k)

    # Store the backward function
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x} if x.requires_grad else set()
    
    return out