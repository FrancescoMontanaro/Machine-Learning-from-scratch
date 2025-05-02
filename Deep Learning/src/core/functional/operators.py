import numpy as np
from typing import Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING: from ..tensor import Tensor
from ..utils.data_analysis import unbroadcast
from .base import tensor_unary_op, tensor_binary_op, accumulate_gradient

# Import the necessary kernel functions
from .kernel.add import add_forward, add_gradient
from .kernel.sub import sub_forward, sub_gradient
from .kernel.mul import mul_forward, mul_gradient
from .kernel.div import div_forward, div_gradient
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
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Compute the sum of the two tensors
        return add_forward(a.data, b.data)
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Compute the gradients for the inputs
        grad_a, grad_b = add_gradient(out_grad, a.data.shape, b.data.shape)

        # Check if the tensors require gradient computation
        if a.requires_grad:
            # Accumulate the gradient of the current tensor
            accumulate_gradient(a, grad_a)

        # Check if the other tensor requires gradient computation
        if b.requires_grad:
            # Accumulate the gradient of the other tensor
            accumulate_gradient(b, grad_b)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op((a, b), forward, backward)


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
    def forward() -> np.ndarray:
        # Perform the subtraction operation
        return sub_forward(a.data, b.data)
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Compute the gradients for the inputs
        grad_a, grad_b = sub_gradient(out_grad, a.data.shape, b.data.shape)
        
        # Check if the tensors require gradient computation
        if a.requires_grad:
            # Accumulate the gradient of the current tensor
            accumulate_gradient(a, grad_a)
            
        # Check if the other tensor requires gradient computation
        if b.requires_grad:
            # Accumulate the gradient of the other tensor
            accumulate_gradient(b, grad_b)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op((a, b), forward, backward)


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
    def forward() -> np.ndarray:
        # Compute the product of the two tensors
        return mul_forward(a.data, b.data)
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Compute the gradients for the inputs
        grad_a, grad_b = mul_gradient(out_grad, a.data, b.data)
        
        # Check if the tensors require gradient computation
        if a.requires_grad:
            # Accumulate the gradient of the current tensor
            accumulate_gradient(a, grad_a)
            
        # Check if the other tensor requires gradient computation
        if b.requires_grad:
            # Accumulate the gradient of the other tensor
            accumulate_gradient(b, grad_b)
            
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op((a, b), forward, backward)


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
    def forward() -> np.ndarray:
        # Compute the division of the two tensors
        return div_forward(a.data, b.data)
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Compute the gradients for the inputs
        grad_a, grad_b = div_gradient(out_grad, a.data, b.data)
        
        # Check if the tensors require gradient computation
        if a.requires_grad:
            # Accumulate the gradient of the current tensor
            accumulate_gradient(a, grad_a)
            
        # Check if the other tensor requires gradient computation
        if b.requires_grad:
            # Accumulate the gradient of the other tensor
            accumulate_gradient(b, grad_b)
            
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op((a, b), forward, backward)


def mat_mul(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to perform matrix multiplication between two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Matrix product of the two tensors
    """
    
    # Define the output data
    out_data: np.ndarray
    
    # Define the forward function
    def forward() -> np.ndarray:
        # Set the nonlocal variable out_data to store the output data
        nonlocal out_data
        
        # Compute the matrix product of the two tensors
        out_data = np.matmul(a.data, b.data)
        
        # Return the output data
        return out_data
    
    # Define the backward function
    def backward(out_grad: np.ndarray) -> None:
        # Check if the tensors require gradient computation
        if a.requires_grad:
            # If the shapes are different, unbroadcast the gradient
            grad_a = np.matmul(out_grad, np.swapaxes(b.data, -1, -2))
            if a.data.shape != out_data.shape:
                # Unbroadcast the gradient
                grad_a = unbroadcast(grad_a, a.data.shape)
                
            # Accumulate the gradient of the current tensor
            accumulate_gradient(a, grad_a)
            
        # Check if the other tensor requires gradient computation
        if b.requires_grad:
            # If the shapes are different, unbroadcast the gradient
            grad_b = np.matmul(np.swapaxes(a.data, -1, -2), out_grad)
            if b.data.shape != out_data.shape:
                # Unbroadcast the gradient
                grad_b = unbroadcast(grad_b, b.data.shape)
                
            # Update the gradient of the other tensor
            accumulate_gradient(b, grad_b)
            
    # Return the tensor operation with the specified forward and backward functions
    return tensor_binary_op((a, b), forward, backward)


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