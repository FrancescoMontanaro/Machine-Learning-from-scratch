import numpy as np
from typing import Callable, Type, Optional, List, Any, TYPE_CHECKING, cast

if TYPE_CHECKING: from ..tensor import Tensor
from ..utils.context_manager import _NO_GRAD
from ..utils.types_registry import get_tensor_class

# Define a global variable to store the tensor class
TensorCls: Optional[Type['Tensor']] = None


class Context:
    
    ### Class attributes ###
    
    # Define the class variable __slots__ to optimize memory usage
    __slots__ = ("_storage",)
    
    ### Magic methods ###

    def __init__(self) -> None:
        """
        Initialize the Context object.
        """
        
        # Initialize the _storage attribute as an empty dictionary
        object.__setattr__(self, "_storage", {})


    def __getattr__(self, name: str) -> Any:
        """
        Retrieve an attribute from the context.
        
        Parameters:
        - name (str): Name of the attribute to retrieve.
        
        Returns:
        - Any: Value of the attribute.
        
        Raises:
        - AttributeError: If the attribute is not found in the context.
        """
        
        # Get the storage dictionary from the object
        storage = object.__getattribute__(self, "_storage")
        
        try:
            # Retrieve the attribute from the storage dictionary
            return storage[name]
        
        ## If the attribute is not found, raise an AttributeError
        except KeyError as err:
            # Raise an AttributeError with a custom message
            raise AttributeError(f"{name!r} not found in Context") from err


    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute in the context.
        
        Parameters:
        - name (str): Name of the attribute to set.
        - value (Any): Value to set for the attribute.
        """
        
        # Check if the attribute name is "_storage"
        if name == "_storage":
            # If it is, set it directly
            object.__setattr__(self, name, value)
        else:
            # Get the storage dictionary from the object
            storage = object.__getattribute__(self, "_storage")
            
            # Set the attribute in the storage dictionary
            storage[name] = value


    ### Public methods ###

    def save(self, **kwargs: Any) -> None:
        """
        Save attributes in the context.
        
        Parameters:
        - kwargs (Any): Attributes to save in the context.
        """
        
        # Get the storage dictionary from the object
        storage = object.__getattribute__(self, "_storage")
        
        # Update the storage dictionary with the provided attributes
        storage.update(kwargs)


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
    global TensorCls
    if TensorCls is None:
        # Lazy load, only the first time
        TensorCls = get_tensor_class()
    
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


def tensor_unary_op_1(t: 'Tensor', forward_fn: Callable[..., np.ndarray], backward_fn: Callable[..., None]) -> 'Tensor':
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
    global TensorCls
    if TensorCls is None:
        # Lazy load, only the first time
        TensorCls = get_tensor_class()
    
    # Check if the input is a tensor
    if not isinstance(t, TensorCls):
        raise TypeError("Input must be an instance of Tensor.")
    
    # Create a context
    ctx = Context()
    
    # Call the forward function
    out_data = forward_fn(ctx, t.data)
    
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
        backward_fn(ctx, out_grad=out.grad, out_buffer=t.grad)
        
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
    global TensorCls
    if TensorCls is None:
        # Lazy load, only the first time
        TensorCls = get_tensor_class()
    
    # Check if the inputs are tensors
    if not isinstance(t1, TensorCls) or not isinstance(t2, TensorCls):
        raise TypeError("All inputs must be instances of Tensor.")
    
    # Create a context
    ctx = Context()
    
    # Call the forward function
    out_data = forward_fn(ctx, t1.data, t2.data)
    
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
            backward_fn_a(ctx, out.grad, t1.grad)
            
        # Check if the second tensor requires gradients
        if t2.requires_grad:
            # If the gradient is None, initialize it to zero
            if t2.grad is None:
                t2.grad = np.zeros_like(t2.data, dtype=t2.data.dtype)
                
            # Backward pass for the second tensor
            backward_fn_b(ctx, out.grad, t2.grad)
        
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
    global TensorCls
    if TensorCls is None:
        # Lazy load, only the first time
        TensorCls = get_tensor_class()

    # Check if the inputs are tensors
    if not all(isinstance(t, TensorCls) for t in tensors):
        raise TypeError("All inputs must be instances of Tensor.")
    
    # Create a context
    ctx = Context()
    
    # Call the forward function
    out_data = forward_fn(ctx, [t.data for t in tensors])
    
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
        
        # Save the index in the context
        ctx.save(idx=0)
        
        # Iterate over the input tensors
        for idx, t in enumerate(tensors):
            # Check if the tensor requires gradients
            if t.requires_grad:
                # If the gradient is None, initialize it to zero
                if t.grad is None:
                    # Initialize the gradient to zero
                    t.grad = np.zeros_like(t.data, dtype=t.data.dtype)
                    
                # Set the context index
                ctx.idx = idx
                
                # Call the backward function with the output gradient
                backward_fn(ctx, out_grad=out.grad, out_buffer=t.grad)
        
    # Set the backward function to the output tensor
    out._backward = _backward
    
    # Set the previous tensors that require gradients
    out._prev = {t for t in tensors if t.requires_grad}
    
    # Return the output tensor
    return out