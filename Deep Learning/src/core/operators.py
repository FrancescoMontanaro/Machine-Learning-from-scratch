import numpy as np
from typing import Union, Type, TYPE_CHECKING, cast


from ..utils import unbroadcast
from .registry import get_tensor_class
if TYPE_CHECKING: from .tensor import Tensor


def add(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to add two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Sum of the two tensors
    
    Raises:
    - AssertionError: If the inputs are not tensors
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Ensure the inputs are tensors
    assert isinstance(a, Tensor) and isinstance(b, Tensor), "Both inputs must be tensors"
    
    # Compute the sum of the two tensors
    out = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    def _backward() -> None:
        if a.requires_grad and out.grad is not None:
            # If the shapes are different, unbroadcast the gradient
            grad_a = out.grad
            if a.data.shape != out.data.shape:
                grad_a = unbroadcast(grad_a, a.data.shape)
                
            # Update the gradient of the current tensor
            a.grad = a.grad + grad_a if a.grad is not None else grad_a
            
        if b.requires_grad and out.grad is not None:
            # If the shapes are different, unbroadcast the gradient
            grad_other = out.grad
            if b.data.shape != out.data.shape:
                grad_other = unbroadcast(grad_other, b.data.shape)
                
            # Update the gradient of the other tensor
            b.grad = b.grad + grad_other if b.grad is not None else grad_other
            
    # Store the backward function with respect to the sum operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {a, b}
    
    # Return the output tensor
    return out


def sub(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to subtract two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Difference of the two tensors
    
    Raises:
    - AssertionError: If the inputs are not tensors
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Ensure the inputs are tensors
    assert isinstance(a, Tensor) and isinstance(b, Tensor), "Both inputs must be tensors"
    
    # Compute the difference of the two tensors
    out = Tensor(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if a.requires_grad and out.grad is not None:
            # Compute the gradient of the loss with respect to the current tensor
            a.grad = a.grad + out.grad if a.grad is not None else out.grad
        if b.requires_grad and out.grad is not None:
            # Compute the gradient of the loss with respect to the other tensor
            b.grad = b.grad - out.grad if b.grad is not None else -out.grad
            
    # Store the backward function with respect to the subtraction operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {a, b}
    
    # Return the output tensor
    return out


def mul(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to multiply two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Product of the two tensors
    
    Raises:
    - AssertionError: If the inputs are not tensors
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Ensure the inputs are tensors
    assert isinstance(a, Tensor) and isinstance(b, Tensor), "Both inputs must be tensors"
    
    # Compute the product of the two tensors
    out = Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if a.requires_grad:
            # Compute the gradient of the loss with respect to the current tensor
            a.grad = a.grad + b.data * out.grad if a.grad is not None else b.data * out.grad
        if b.requires_grad:
            # Compute the gradient of the loss with respect to the other tensor
            b.grad = b.grad + a.data * out.grad if b.grad is not None else a.data * out.grad
            
    # Store the backward function with respect to the product operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {a, b}
    
    # Return the output tensor
    return out


def div(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to divide two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Quotient of the two tensors
    
    Raises:
    - AssertionError: If the inputs are not tensors
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Ensure the inputs are tensors
    assert isinstance(a, Tensor) and isinstance(b, Tensor), "Both inputs must be tensors"
    
    # Compute the division of the two tensors
    out = Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if a.requires_grad:
            # Compute the gradient of the loss with respect to the current tensor
            a.grad = a.grad + out.grad / b.data if a.grad is not None else out.grad / b.data
        if b.requires_grad:
            # Compute the gradient of the loss with respect to the other tensor
            b.grad = b.grad - a.data * out.grad / (b.data ** 2) if b.grad is not None else -a.data * out.grad / (b.data ** 2)
            
    # Store the backward function with respect to the division operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {a, b}
    
    # Return the output tensor
    return out


def mat_mul(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to perform matrix multiplication between two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Matrix product of the two tensors
    
    Raises:
    - AssertionError: If the inputs are not tensors
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Ensure the inputs are tensors
    assert isinstance(a, Tensor) and isinstance(b, Tensor), "Both inputs must be tensors"
    
    # Compute the matrix multiplication of the two tensors
    out = Tensor(np.matmul(a.data, b.data), requires_grad=a.requires_grad or b.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if a.requires_grad and out.grad is not None:
            # Gradient w.r.t. the current tensor
            grad_self = np.matmul(out.grad, np.swapaxes(b.data, -1, -2))
            
            # Unbroadcast grad_self to match self.data shape
            grad_self = unbroadcast(grad_self, a.data.shape)
            
            # Update the gradient of the current tensor
            a.grad = a.grad + grad_self if a.grad is not None else grad_self
            
        if b.requires_grad and out.grad is not None:
            # Gradient w.r.t. the other tensor
            grad_other = np.matmul(np.swapaxes(a.data, -1, -2), out.grad)
            
            # Unbroadcast grad_other to match other.data shape
            grad_other = unbroadcast(grad_other, b.data.shape)
            
            # Update the gradient of the other tensor
            b.grad = b.grad + grad_other if b.grad is not None else grad_other
            
    # Store the backward function with respect to the matrix multiplication operation
    out._backward = _backward
        
    # Store the previous tensors in the computation graph
    out._prev = {a, b}
    
    # Return the output tensor
    return out


def radd(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to add two tensors in reverse order.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Sum of the two tensors
    
    Raises:
    - AssertionError: If the inputs are not tensors
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Ensure the inputs are tensors
    assert isinstance(a, Tensor) and isinstance(b, Tensor), "Both inputs must be tensors"
    
    # Compute the sum of the two tensors
    out = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # The derivative of (self + other) w.r.t self is 1, so we propagate out.grad
        if a.requires_grad and out.grad is not None:
            a.grad = a.grad + out.grad if a.grad is not None else out.grad
    
    # Store the backward function with respect to the sum operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {a, b}
    
    # Return the output tensor
    return out


def rsub(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to subtract two tensors in reverse order.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Difference of the two tensors
    
    Raises:
    - AssertionError: If the inputs are not tensors
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Ensure the inputs are tensors
    assert isinstance(a, Tensor) and isinstance(b, Tensor), "Both inputs must be tensors"
    
    # Compute the subtraction
    out = Tensor(b.data - a.data, requires_grad=a.requires_grad or b.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # The derivative of (other - self) w.r.t self is -1, so we propagate -out.grad
        if a.requires_grad and out.grad is not None:
            a.grad = a.grad - out.grad if a.grad is not None else - out.grad
    
    # Store the backward function with respect to the subtraction operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {a, b}
    
    # Return the output tensor
    return out


def rmul(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to multiply two tensors in reverse order.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Product of the two tensors
    
    Raises:
    - AssertionError: If the inputs are not tensors
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Ensure the inputs are tensors
    assert isinstance(a, Tensor) and isinstance(b, Tensor), "Both inputs must be tensors"
    
    # Compute the product of the two tensors
    out = Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # The derivative of (self * other) w.r.t self is other, so we propagate other * out.grad
        if a.requires_grad and out.grad is not None:
            a.grad = a.grad + b.data * out.grad if a.grad is not None else b.data * out.grad
            
    # Store the backward function with respect to the product operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {a, b}
    
    # Return the output tensor
    return out


def rdiv(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to divide two tensors in reverse order.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Quotient of the two tensors
    
    Raises:
    - AssertionError: If the inputs are not tensors
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Ensure the inputs are tensors
    assert isinstance(a, Tensor) and isinstance(b, Tensor), "Both inputs must be tensors"
    
    # Compute the division of the two tensors
    out = Tensor(b.data / a.data, requires_grad=a.requires_grad or b.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # The derivative of (other / self) w.r.t self is -other / (self ** 2), so we propagate -other / (self ** 2) * out.grad
        if a.requires_grad and out.grad is not None:
            a.grad = a.grad - b.data * out.grad / (a.data ** 2) if a.grad is not None else -b.data * out.grad / (a.data ** 2)
    
    # Store the backward function with respect to the division operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {a, b}
    
    # Return the output tensor
    return out


def pow(x: 'Tensor', power: Union[int, float]) -> 'Tensor':
    """
    Function to raise a tensor to the power of another tensor.
    
    Parameters:
    - x (Tensor): Base tensor
    - power (Union[int, float]): Exponent tensor
    
    Returns:
    - Tensor: Base tensor raised to the power of the exponent tensor
    
    Raises:
    - AssertionError: If the inputs are not tensors
    - AssertionError: If the power is not an integer or a float
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Ensure the inputs are tensors
    assert isinstance(x, Tensor), "The input must be a tensor"
    assert isinstance(power, (int, float)), "The power must be an integer or a float"
    
    # Compute the power of the tensor
    out = Tensor(x.data ** power, requires_grad=x.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Compute the gradient of the loss with respect to the current tensor
            grad_self = power * (x.data ** (power - 1)) * out.grad
            
            # Update the gradient of the current tensor
            x.grad = x.grad + grad_self if x.grad is not None else grad_self
    
    # Store the backward function with respect to the power operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x}
    
    # Return the output tensor
    return out


def neg(x: 'Tensor') -> 'Tensor':
    """
    Function to negate a tensor.
    
    Parameters:
    - x (Tensor): Input tensor
    
    Returns:
    - Tensor: Negated tensor
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Ensure the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Compute the negation of the tensor
    out = Tensor(-x.data, requires_grad=x.requires_grad)
    
    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Compute the gradient of the loss with respect to the current tensor
            x.grad = x.grad - out.grad if x.grad is not None else -out.grad
    
    # Store the backward function with respect to the negation operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x}
    
    # Return the output tensor
    return out


def get_item(x: 'Tensor', key: Union[int, slice, np.ndarray]) -> 'Tensor':
    """
    Function to slice a tensor.
    
    Parameters:
    - x (Tensor): Input tensor
    - key (Union[int, slice, np.ndarray]): Index or slice object
    
    Returns:
    - Tensor: Sliced tensor
    
    Raises:
    - AssertionError: If the input is not a tensor
    """
    
    # Get the tensor class
    Tensor = cast(Type['Tensor'], get_tensor_class())
    
    # Ensure the input is a tensor
    assert isinstance(x, Tensor), "Input must be a tensor"
    
    # Get the sliced data from the underlying NumPy array.
    out = Tensor(x.data[key], requires_grad=x.requires_grad)

    # Define the backward function
    def _backward() -> None:
        # If the gradient needs to be computed, backpropagate the gradient
        if x.requires_grad and out.grad is not None:
            # Create an array of zeros with the same shape as the input tensor
            grad_self = np.zeros_like(x.data)
            
            # Scatter the gradient from out.grad into grad_self at the positions specified by key.
            np.add.at(grad_self, key, out.grad) # type: ignore
            
            # Accumulate the gradient into self.grad.
            x.grad = x.grad + grad_self if x.grad is not None else grad_self

    # Store the backward function with respect to the negation operation
    out._backward = _backward
    
    # Store the previous tensors in the computation graph
    out._prev = {x}
    
    # Return the output tensor
    return out