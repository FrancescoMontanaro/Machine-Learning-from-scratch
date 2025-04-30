import numpy as np
from typing import TYPE_CHECKING

from .base import tensor_op
if TYPE_CHECKING: from ..tensor import Tensor

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
    """
    
    # Extract the number of elements in the input tensor and create an empty array to store the output data
    n = x.data.size
    out_data: np.ndarray
    
    # Define the forward function
    def forward(data: np.ndarray) -> np.ndarray:
        # Set the nonlocal variable out_data to store the output data
        nonlocal out_data
        
        # Create an empty array to store the output data
        out_data = np.empty_like(data)
        
        # Compute the sigmoid function
        sigmoid_forward(data.ravel(), out_data.ravel(), n)
        
        # Return the output data
        return out_data
    
    def backward(t: 'Tensor', out_grad: np.ndarray) -> None:
        # Check if the tensor requires gradient computation
        if not t.requires_grad:
            return
        
        # Check if the gradient is initialized
        assert t.grad is not None, "Gradient must be initialized"
        
        # Compute the gradient of the sigmoid function
        sigmoid_gradient(out_data.ravel(), out_grad.ravel(), t.grad.ravel(), n)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_op(x, forward, backward)


def relu(x: 'Tensor') -> 'Tensor':
    """
    Compute the ReLU activation function.

    Parameters:
    - x (Tensor): Input tensor

    Returns:
    - Tensor: Output tensor
    """
    
    # Extract the number of elements in the input tensor and create an empty array to store the output data
    n = x.data.size
    out_data: np.ndarray
    
    # Define the forward function
    def forward(data: np.ndarray) -> np.ndarray:
        # Set the nonlocal variable out_data to store the output data
        nonlocal out_data
        
        # Create an empty array to store the output data
        out_data = np.empty_like(data)
        
        # Compute the ReLU function
        relu_forward(data.ravel(), out_data.ravel(), n)
        
        # Return the output data
        return out_data
    
    # Define the backward function
    def backward(t: 'Tensor', out_grad: np.ndarray) -> None:
        # Check if the tensor requires gradient computation
        if not t.requires_grad:
            return
        
        # Check if the gradient is initialized
        assert t.grad is not None, "Gradient must be initialized"
        
        # Compute the gradient of the ReLU function
        relu_gradient(out_data.ravel(), out_grad.ravel(), t.grad.ravel(), n)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_op(x, forward, backward)


def tanh(x: 'Tensor') -> 'Tensor':
    """
    Compute the tanh activation function.

    Parameters:
    - x (Tensor): Input tensor

    Returns:
    - Tensor: Output tensor
    """
    
    # Extract the number of elements in the input tensor and create an empty array to store the output data
    n = x.data.size
    out_data: np.ndarray
    
    # Define the forward function
    def forward(data: np.ndarray) -> np.ndarray:
        # Set the nonlocal variable out_data to store the output data
        nonlocal out_data
        
        # Create an empty array to store the output data
        out_data = np.empty_like(data)
        
        # Compute the tanh function
        tanh_forward(data.ravel(), out_data.ravel(), n)
        
        # Return the output data
        return out_data
    
    # Define the backward function
    def backward(t: 'Tensor', out_grad: np.ndarray) -> None:
        # Check if the tensor requires gradient computation
        if not t.requires_grad:
            return
        
        # Check if the gradient is initialized
        assert t.grad is not None, "Gradient must be initialized"
        
        # Compute the gradient of the tanh function
        tanh_gradient(out_data.ravel(), out_grad.ravel(), t.grad.ravel(), n)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_op(x, forward, backward)


def softmax(x: 'Tensor', axis: int = -1) -> 'Tensor':
    """
    Compute the softmax activation function.

    Parameters:
    - x (Tensor): Input tensor

    Returns:
    - Tensor: Output tensor
    """
    
    # Extract the number of elements in the input tensor and create an empty array to store the output data
    k = x.data.shape[-1]
    n = x.data.size // k
    out_data: np.ndarray
    
    # Define the forward function
    def forward(data: np.ndarray) -> np.ndarray:
        # Set the nonlocal variable out_data to store the output data
        nonlocal out_data
        
        # Extract number of elements in the input tensor
        ndim = data.ndim
        
        # Compute the axis for softmax
        ax = axis % ndim
        
        # If the axis is not the last one, compute softmax along the specified axis
        if ax != ndim - 1:
            # Compute the maximum value along the specified axis
            out_data = np.exp(data) / np.sum(np.exp(data), axis=ax, keepdims=True)
        # If the axis is the last one, compute softmax using the kernel function
        else:
            # Create an empty array to store the output data
            out_data = np.empty_like(data)
        
            # Compute the softmax function
            softmax_forward(data.ravel(), out_data.ravel(), n, k)
        
        # Return the output data
        return out_data
    
    # Define the backward function
    def backward(t: 'Tensor', out_grad: np.ndarray) -> None:
        # Check if the tensor requires gradient computation
        if not t.requires_grad:
            return
        
        # Check if the gradient is initialized
        assert t.grad is not None, "Gradient must be initialized"
        
        # Compute the gradient of the softmax function
        softmax_gradient(out_data.ravel(), out_grad.ravel(), t.grad.ravel(), n, k)
        
    # Return the tensor operation with the specified forward and backward functions
    return tensor_op(x, forward, backward)


def log_softmax(x: 'Tensor', axis: int = -1) -> 'Tensor':
    """
    Compute the log softmax activation function.

    Parameters:
    - x (Tensor): Input tensor
    - axis (int): Axis along which to compute softmax. Default: -1 (last axis)

    Returns:
    - Tensor: Output tensor
    """
    
    # Extract the number of elements in the input tensor and create an empty array to store the output data
    k = x.data.shape[-1]
    n = x.data.size // k
    out_data: np.ndarray
    
    # Define the forward function
    def forward(data: np.ndarray) -> np.ndarray:
        # Set the nonlocal variable out_data to store the output data
        nonlocal out_data
        
        # Extract the number of elements in the input tensor
        ndim = data.ndim
        
        # Compute the axis for log softmax
        ax = axis % ndim
        
        # If the axis is not the last one, compute log softmax along the specified axis
        if ax != ndim - 1:
            # Compute the maximum value along the specified axis
            m = np.max(data, axis=axis, keepdims=True)
            
            # Subtract the maximum value from the input data
            y = data - m
            
            # Compute the log sum of exponentials
            logsum = np.log(np.sum(np.exp(y), axis=axis, keepdims=True))
            
            # Compute the log softmax
            out_data = y - logsum
        
        # If the axis is the last one, compute log softmax using the kernel function
        else:
            # Create an empty array to store the output data
            out_data = np.empty_like(data)
            
            # Compute the log softmax function
            log_softmax_forward(data.ravel(), out_data.ravel(), n, k)
            
        # Return the output data
        return out_data
    
    # Define the backward function
    def backward(t: 'Tensor', out_grad: np.ndarray) -> None:
        # Check if the tensor requires gradient computation
        if not t.requires_grad:
            return
        
        # Check if the gradient is initialized
        assert t.grad is not None, "Gradient must be initialized"
        
        # Compute the gradient of the log softmax function
        log_softmax_gradient(out_data.ravel(), out_grad.ravel(), t.grad.ravel(), n, k)
        
    # Return the tensor operation with the specified forward and backward functions
    return tensor_op(x, forward, backward)