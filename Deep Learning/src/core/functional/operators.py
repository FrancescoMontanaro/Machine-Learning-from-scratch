import numpy as np
from typing import Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING: from ..tensor import Tensor
from ..utils.data_analysis import unbroadcast
from .base import tensor_op, tensor_tuple_op, accumulate_gradient


def add(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to add two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Sum of the two tensors
    """
    
    # Define the output data
    out_data: np.ndarray
    
    # Define the forward function
    def forward(data: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        # Set the nonlocal variable out_data to store the output data
        nonlocal out_data
        
        # Unpack the input data
        a_data, b_data = data
        
        # Compute the sum of the two tensors
        out_data = a_data + b_data
        
        # Return the output data
        return out_data
    
    # Define the backward function
    def backward(input_tuple: tuple['Tensor', 'Tensor'], out_grad: np.ndarray) -> None:
        # Unpack the input tensors
        a, b = input_tuple
        
        # Check if the tensors require gradient computation
        if a.requires_grad:
            # If the shapes are different, unbroadcast the gradient
            grad_a = out_grad
            if a.data.shape != out_data.shape:
                grad_a = unbroadcast(grad_a, a.data.shape)
                
            # Accumulate the gradient of the current tensor
            accumulate_gradient(a, grad_a)
            
        # Check if the other tensor requires gradient computation
        if b.requires_grad:
            # If the shapes are different, unbroadcast the gradient
            grad_b = out_grad
            if b.data.shape != out_data.shape:
                grad_b = unbroadcast(grad_b, b.data.shape)
                
            # Update the gradient of the other tensor
            accumulate_gradient(b, grad_b)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_tuple_op((a, b), forward, backward)


def sub(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to subtract two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Difference of the two tensors
    """
    
    # Define the output data
    out_data: np.ndarray
    
    # Define the forward function
    def forward(data: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        # Set the nonlocal variable out_data to store the output data
        nonlocal out_data
        
        # Unpack the input data
        a_data, b_data = data
        
        # Compute the difference of the two tensors
        out_data = a_data - b_data
        
        # Return the output data
        return out_data
    
    # Define the backward function
    def backward(input_tuple: tuple['Tensor', 'Tensor'], out_grad: np.ndarray) -> None:
        # Unpack the input tensors
        a, b = input_tuple
        
        # Check if the tensors require gradient computation
        if a.requires_grad:
            # If the shapes are different, unbroadcast the gradient
            grad_a = out_grad
            if a.data.shape != out_data.shape:
                grad_a = unbroadcast(grad_a, a.data.shape)
                
            # Accumulate the gradient of the current tensor
            accumulate_gradient(a, grad_a)
            
        # Check if the other tensor requires gradient computation
        if b.requires_grad:
            # If the shapes are different, unbroadcast the gradient
            grad_b = -out_grad
            if b.data.shape != out_data.shape:
                grad_b = unbroadcast(grad_b, b.data.shape)
                
            # Update the gradient of the other tensor
            accumulate_gradient(b, grad_b)
    
    # Return the tensor operation with the specified forward and backward functions
    return tensor_tuple_op((a, b), forward, backward)


def mul(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to multiply two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Product of the two tensors
    """
    
    # Define the output data
    out_data: np.ndarray
    
    # Define the forward function
    def forward(data: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        # Set the nonlocal variable out_data to store the output data
        nonlocal out_data
        
        # Unpack the input data
        a_data, b_data = data
        
        # Compute the product of the two tensors
        out_data = a_data * b_data
        
        # Return the output data
        return out_data
    
    # Define the backward function
    def backward(input_tuple: tuple['Tensor', 'Tensor'], out_grad: np.ndarray) -> None:
        # Unpack the input tensors
        a, b = input_tuple
        
        # Check if the tensors require gradient computation
        if a.requires_grad:
            # If the shapes are different, unbroadcast the gradient
            grad_a = b.data * out_grad
            if a.data.shape != out_data.shape:
                # Unbroadcast the gradient
                grad_a = unbroadcast(grad_a, a.data.shape)
                
            # Accumulate the gradient of the current tensor
            accumulate_gradient(a, grad_a)
            
        # Check if the other tensor requires gradient computation
        if b.requires_grad:
            # If the shapes are different, unbroadcast the gradient
            grad_b = a.data * out_grad
            if b.data.shape != out_data.shape:
                # Unbroadcast the gradient
                grad_b = unbroadcast(grad_b, b.data.shape)
                
            # Update the gradient of the other tensor
            accumulate_gradient(b, grad_b)
            
    # Return the tensor operation with the specified forward and backward functions
    return tensor_tuple_op((a, b), forward, backward)


def div(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Function to divide two tensors.
    
    Parameters:
    - a (Tensor): First tensor
    - b (Tensor): Second tensor
    
    Returns:
    - Tensor: Quotient of the two tensors
    """
    
    # Define the output data
    out_data: np.ndarray
    
    # Define the forward function
    def forward(data: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        # Set the nonlocal variable out_data to store the output data
        nonlocal out_data
        
        # Unpack the input data
        a_data, b_data = data
        
        # Compute the quotient of the two tensors
        out_data = a_data / b_data
        
        # Return the output data
        return out_data
    
    # Define the backward function
    def backward(input_tuple: tuple['Tensor', 'Tensor'], out_grad: np.ndarray) -> None:
        # Unpack the input tensors
        a, b = input_tuple
        
        # Check if the tensors require gradient computation
        if a.requires_grad:
            # If the shapes are different, unbroadcast the gradient
            grad_a = out_grad / b.data
            if a.data.shape != out_data.shape:
                # Unbroadcast the gradient
                grad_a = unbroadcast(grad_a, a.data.shape)
                
            # Accumulate the gradient of the current tensor
            accumulate_gradient(a, grad_a)
            
        # Check if the other tensor requires gradient computation
        if b.requires_grad:
            # If the shapes are different, unbroadcast the gradient
            grad_b = -a.data * out_grad / (b.data ** 2)
            if b.data.shape != out_data.shape:
                # Unbroadcast the gradient
                grad_b = unbroadcast(grad_b, b.data.shape)
                
            # Update the gradient of the other tensor
            accumulate_gradient(b, grad_b)
            
    # Return the tensor operation with the specified forward and backward functions
    return tensor_tuple_op((a, b), forward, backward)


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
    def forward(data: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        # Set the nonlocal variable out_data to store the output data
        nonlocal out_data
        
        # Unpack the input data
        a_data, b_data = data
        
        # Compute the matrix product of the two tensors
        out_data = np.matmul(a_data, b_data)
        
        # Return the output data
        return out_data
    
    # Define the backward function
    def backward(input_tuple: tuple['Tensor', 'Tensor'], out_grad: np.ndarray) -> None:
        # Unpack the input tensors
        a, b = input_tuple
        
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
    return tensor_tuple_op((a, b), forward, backward)


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
    def forward(data: np.ndarray) -> np.ndarray:
        # Compute the power of the tensor
        return data ** power
    
    # Define the backward function
    def backward(t: 'Tensor', out_grad: np.ndarray) -> None:
        # Check if the tensor requires gradient computation
        if not t.requires_grad:
            return
        
        # Check if the gradient is initialized
        assert t.grad is not None, "Gradient must be initialized"
        
        # Compute the gradient of the power function
        grad = power * (t.data ** (power - 1)) * out_grad
        
        # Accumulate the gradient into self.grad.
        accumulate_gradient(t, grad)
        
    # Return the tensor operation with the specified forward and backward functions
    return tensor_op(x, forward, backward)


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
    def forward(data: np.ndarray) -> np.ndarray:
        # Return the sliced data
        return data[key]
    
    # Define the backward function
    def backward(t: 'Tensor', out_grad: np.ndarray) -> None:
        # Check if the tensor requires gradient computation
        if not t.requires_grad:
            return
        
        # Check if the gradient is initialized
        assert t.grad is not None, "Gradient must be initialized"
        
        # Compute the gradient of the slice operation
        grad = np.zeros_like(t.data)
        
        # Use direct assignment which handles slices correctly
        np.add.at(grad, key, out_grad) # type: ignore
                
        # Accumulate the gradient into self.grad.
        accumulate_gradient(t, grad)
        
    # Return the tensor operation with the specified forward and backward functions
    return tensor_op(x, forward, backward)