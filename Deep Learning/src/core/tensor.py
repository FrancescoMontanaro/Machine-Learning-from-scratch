import weakref
import numpy as np
from types import EllipsisType
from typing import Optional, Union, Tuple, List, Callable, Any

from .utils import _noop
from .functional.kernel import *
from .functional.tape import tape_push, tape_clear
from .functional.base import (
    tensor_unary_op, 
    tensor_binary_op, 
    tensor_nary_op, 
    tensor_unary_op_multiple_outputs,
    tensor_unary_op_binary_output
)


class Tensor:
    
    ##########################
    ##### Class variables ####
    ##########################
    
    # Class variable to keep track of live tensors
    _live_tensors = weakref.WeakSet()
    __slots__ = ("data", "is_parameter", "requires_grad", "grad", "_backward", "_prev", "__weakref__")
    
    
    ###########################
    ###### Magic methods ######
    ###########################
    
    def __init__(self, data: Union[int, float, np.ndarray], requires_grad: bool = False, dtype: type = np.float32, is_parameter: bool = False) -> None:
        """
        Constructor for the Tensor class
        
        Parameters:
        - data (np.ndarray): Data of the tensor (numpy array)
        - requires_grad (bool): Flag to indicate if the gradient needs to be computed for the tensor
        - dtype (type): Data type of the tensor. Default is np.float32
        - is_parameter (bool): Flag to indicate if the tensor is a trainable parameter
        """
        
        # Save the raw data
        self.data = np.asarray(data, dtype=dtype)
        
        # Gradient tracking flags
        self.is_parameter = is_parameter
        self.requires_grad = requires_grad
        
        # Initialize gradient and graph metadata
        self.grad = None
        self._backward: Callable = _noop  # Use static function, not lambda
        self._prev: set['Tensor'] = set()

        # Register live tensors
        Tensor._live_tensors.add(self)


    def __repr__(self) -> str:
        """
        Method to return the string representation of the tensor
        
        Returns:
        - str: String representation of the tensor
        """
        
        # Return the string representation of the tensor
        return f"Tensor({self.data}, shape={self.data.shape}, dtype={self.data.dtype})"


    #############################
    ### Operators overloading ###
    #############################

    def __add__(self, other: Union[int, float, np.ndarray, 'Tensor']) -> 'Tensor':
        """
        Method to add two tensors
        
        Parameters:
        - other (Union[int, float, np.ndarray, Tensor]): Object to be added to the current tensor
        
        Returns:
        - Tensor: Tensor containing the sum of the two tensors
        """

        # Check if the object is an integer or a float
        if not isinstance(other, Tensor):
            # Convert the integer or float to a tensor
            other = Tensor(
                data = np.array(other, dtype=self.data.dtype) if isinstance(other, (int, float)) else other, 
                requires_grad = False
            )
        
        # Capture shapes locally to avoid closure over entire tensors
        shape_a = self.data.shape
        shape_b = other.data.shape
            
        # Define the forward function
        def forward(a_data: np.ndarray, b_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the sum of the two tensors (no need to save data for add backward)
            return add_forward(a_data, b_data), -1
        
        # Define the backward function for tensor a
        def backward_a(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the addition operation using captured shape
            add_backward(out_grad=out_grad, out_buffer=out_buffer, target_shape=shape_a)
            
        # Define the backward function for tensor b
        def backward_b(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the addition operation using captured shape
            add_backward(out_grad=out_grad, out_buffer=out_buffer, target_shape=shape_b)
                
        # Return the tensor operation with the specified forward and backward functions
        return tensor_binary_op(
            t1 = self,
            t2 = other,
            forward_fn = forward,
            backward_fn_a = backward_a,
            backward_fn_b = backward_b,
            tensor_cls = Tensor
        )
    
    
    def __sub__(self, other: Union[int, float, np.ndarray, 'Tensor']) -> 'Tensor':
        """
        Method to subtract two tensors
        
        Parameters:
        - other (Union[int, float, np.ndarray, Tensor]): Object to be subtracted from the current tensor
        
        Returns:
        - Tensor: Tensor containing the difference of the two tensors
        """
        
        # Check if the object is an integer or a float
        if not isinstance(other, Tensor):
            # Convert the integer or float to a tensor
            other = Tensor(
                data = np.array(other, dtype=self.data.dtype) if isinstance(other, (int, float)) else other, 
                requires_grad = False
            )
        
        # Capture shapes locally to avoid closure over entire tensors
        shape_a = self.data.shape
        shape_b = other.data.shape
            
        # Define the forward function
        def forward(a_data: np.ndarray, b_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the difference of the two tensors (no need to save data for sub backward)
            return sub_forward(a_data, b_data), -1
        
        # Define the backward function for tensor a
        def backward_a(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the subtraction operation using captured shape
            sub_backward_a(out_grad=out_grad, out_buffer=out_buffer, target_shape=shape_a)
            
        # Define the backward function for tensor b
        def backward_b(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the subtraction operation using captured shape
            sub_backward_b(out_grad=out_grad, out_buffer=out_buffer, target_shape=shape_b)
                
        # Return the tensor operation with the specified forward and backward functions
        return tensor_binary_op(
            t1 = self,
            t2 = other,
            forward_fn = forward,
            backward_fn_a = backward_a,
            backward_fn_b = backward_b,
            tensor_cls = Tensor
        )
    

    def __mul__(self, other: Union[int, float, np.ndarray, 'Tensor']) -> 'Tensor':
        """
        Method to multiply two tensors
        
        Parameters:
        - other (Union[int, float, Tensor]): Object to be multiplied with the current tensor
        
        Returns:
        - Tensor: Tensor containing the product of the two tensors
        """
        
        # Check if the object is an integer or a float
        if not isinstance(other, Tensor):
            # Convert the integer or float to a tensor
            other = Tensor(
                data = np.array(other, dtype=self.data.dtype) if isinstance(other, (int, float)) else other, 
                requires_grad = False
            )
        
        # Capture shapes locally to avoid closure over entire tensors
        shape_a = self.data.shape
        shape_b = other.data.shape
            
        # Define the forward function - save input data in tape for backward
        def forward(a_data: np.ndarray, b_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the product and save both inputs for backward pass
            return mul_forward(a_data, b_data), tape_push((a_data, b_data))
        
        # Define the backward function for tensor a
        def backward_a(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:
            # Retrieve b_data from saved_data
            _, b_data = saved_data if saved_data else (None, None)
            if b_data is not None:
                mul_backward_a(out_grad=out_grad, out_buffer=out_buffer, target_shape=shape_a, b_data=b_data)
            
        # Define the backward function for tensor b
        def backward_b(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:
            # Retrieve a_data from saved_data
            a_data, _ = saved_data if saved_data else (None, None)
            if a_data is not None:
                mul_backward_b(out_grad=out_grad, out_buffer=out_buffer, target_shape=shape_b, a_data=a_data)
                
        # Return the tensor operation with the specified forward and backward functions
        return tensor_binary_op(
            t1 = self,
            t2 = other,
            forward_fn = forward,
            backward_fn_a = backward_a,
            backward_fn_b = backward_b,
            tensor_cls = Tensor
        )
    
    
    def __truediv__(self, other: Union[int, float, np.ndarray, 'Tensor']) -> 'Tensor':
        """
        Method to divide two tensors
        
        Parameters:
        - other (Union[int, float, np.ndarray, Tensor]): Object to be divided by the current tensor
        
        Returns:
        - Tensor: Tensor containing the division of the two tensors
        """
        
        # Check if the object is an integer or a float
        if not isinstance(other, Tensor):
            # Convert the integer or float to a tensor
            other = Tensor(
                data = np.array(other, dtype=self.data.dtype) if isinstance(other, (int, float)) else other, 
                requires_grad = False
            )
        
        # Capture shapes locally to avoid closure over entire tensors
        shape_a = self.data.shape
            
        # Define the forward function - save input data in tape for backward
        def forward(a_data: np.ndarray, b_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the division and save both inputs for backward pass
            return div_forward(a_data, b_data), tape_push((a_data, b_data))
        
        # Define the backward function for tensor a
        def backward_a(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:
            # Retrieve b_data from saved_data
            _, b_data = saved_data if saved_data else (None, None)
            if b_data is not None:
                div_backward_a(out_grad=out_grad, out_buffer=out_buffer, target_shape=shape_a, b_data=b_data)
            
        # Define the backward function for tensor b
        def backward_b(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:
            # Retrieve both a_data and b_data from saved_data
            a_data, b_data = saved_data if saved_data else (None, None)
            if a_data is not None and b_data is not None:
                div_backward_b(out_grad=out_grad, out_buffer=out_buffer, a_data=a_data, b_data=b_data)
                
        # Return the tensor operation with the specified forward and backward functions
        return tensor_binary_op(
            t1 = self,
            t2 = other,
            forward_fn = forward,
            backward_fn_a = backward_a,
            backward_fn_b = backward_b,
            tensor_cls = Tensor
        )


    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """
        Method to perform matrix multiplication of two tensors
        
        Parameters:
        - other (Tensor): Tensor to be multiplied with the current tensor
        
        Returns:
        - Tensor: Tensor containing the matrix multiplication of the two tensors
        
        Raises:
        - AssertionError: If the object to be multiplied is not a tensor
        """
            
        # Define the forward function - save input data in tape for backward
        def forward(a_data: np.ndarray, b_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the matrix multiplication and save both inputs for backward pass
            return matmul_forward(a_data, b_data), tape_push((a_data, b_data))
        
        # Define the backward function for tensor a
        def backward_a(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:
            # Retrieve b_data from saved_data
            _, b_data = saved_data if saved_data else (None, None)
            if b_data is not None:
                matmul_backward_a(out_grad=out_grad, out_buffer=out_buffer, b_data=b_data)
            
        # Define the backward function for tensor b
        def backward_b(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:
            # Retrieve a_data from saved_data
            a_data, _ = saved_data if saved_data else (None, None)
            if a_data is not None:
                matmul_backward_b(out_grad=out_grad, out_buffer=out_buffer, a_data=a_data)
                
        # Return the tensor operation with the specified forward and backward functions
        return tensor_binary_op(
            t1 = self,
            t2 = other,
            forward_fn = forward,
            backward_fn_a = backward_a,
            backward_fn_b = backward_b,
            tensor_cls = Tensor
        )
    
    
    def __radd__(self, other: Union[int, float, np.ndarray]) -> 'Tensor':
        """
        Method to implement the reverse addition so that expressions like (1 + tensor) can be evaluated.
        
        Parameters:
        - other (Union[int, float, np.ndarray]): A constant value on the left-hand side.
        
        Returns:
        - Tensor: A new Tensor representing (other + self.data).
        """
            
        # Compute the sum of the two tensors
        return self.__add__(other)
    
    
    def __rsub__(self, other: Union[int, float, np.ndarray]) -> 'Tensor':
        """
        Method to implement the reverse subtraction so that expressions like (1 - tensor) can be evaluated.
        
        Parameters:
        - other (int, float, np.ndarray): A constant value on the left-hand side.
        
        Returns:
        - Tensor: A new Tensor representing (other - self.data).
        """
            
        # Compute the difference of the two tensors
        return -self.__sub__(other)
    
    
    def __rmul__(self, other: Union[int, float, np.ndarray]) -> 'Tensor':
        """
        Method to implement the reverse multiplication so that expressions like (2 * tensor) can be evaluated.
        
        Parameters:
        - other (Union[int, float, np.ndarray]): A constant value on the left-hand side.
        
        Returns:
        - Tensor: A new Tensor representing (other * self.data).
        """
        
        # Delegate to __mul__ by reversing the order of the operands
        return self.__mul__(other)
    
    
    def __rtruediv__(self, other: Union[int, float, np.ndarray]) -> 'Tensor':
        """
        Method to implement the reverse division so that expressions like (2 / tensor) can be evaluated.
        
        Parameters:
        - other (Union[int, float, np.ndarray]): A constant value on the left-hand side.
        
        Returns:
        - Tensor: A new Tensor representing (other / self.data).
        """
        
        # Assert that the object to be divided is an integer or a float
        assert isinstance(other, (int, float, np.ndarray)), "The object to be divided should be an integer or a float."
        
        # Convert the constant to a Tensor (with requires_grad=False)
        other_tensor = Tensor(
            data = np.array(other, dtype=self.data.dtype) if isinstance(other, (int, float)) else other, 
            requires_grad = False
        )
            
        # Compute the division of the two tensors
        return other_tensor.__truediv__(self)
    
    
    def __pow__(self, power: Union[int, float]) -> 'Tensor':
        """
        Method to compute the power of the tensor
        
        Parameters:
        - power (Union[int, float]): Power to be raised
        
        Returns:
        - Tensor: Tensor containing the power of the current tensor
        
        Raises:
        - AssertionError: If the power is not an integer or a float
        """
        
        # Ensure the power is a scalar
        assert isinstance(power, (int, float)), "The power must be an integer or a float"
        
        # Define the forward function - save input data in tape for backward
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the power and save input for backward pass
            return pow_forward(x_data, power), tape_push((x_data,))
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:
            # Retrieve x_data from saved_data
            x_data = saved_data[0] if saved_data else None
            if x_data is not None:
                # Compute the gradient of the power function and accumulate in-place
                grad = pow_gradient(out_grad, x_data, power)
                np.add(out_buffer, grad, out=out_buffer)
            
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    def __neg__(self) -> 'Tensor':
        """
        Unary negation operator, returning a new Tensor with negated data.
        
        Returns:
        - Tensor: A new Tensor with negated data.
        """
        
        # Compute and return the negation of the tensor
        return (-1) * self
    
    
    def __getitem__(self, key: Union[int, slice, np.ndarray, EllipsisType, Tuple[Union[int, slice, np.ndarray, EllipsisType], ...]]) -> 'Tensor':
        """
        Implements slicing for the tensor using the [] operator.
        
        Parameters:
        - key (Union[int, slice, np.ndarray, Tuple[Union[int, slice, np.ndarray], ...]]): Key to slice the tensor

        Returns:
        - Tensor: A new tensor with data = self.data[key].
        """
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Return the sliced data (no need to save for backward, shape is captured)
            return x_data[key], -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Use direct assignment which handles slices correctly
            np.add.at(out_buffer, key, out_grad)  # type: ignore
            
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
        
        
    def __setitem__(self, key: Union[int, slice, np.ndarray, Tuple[Union[int, slice, np.ndarray], ...]], value: Union[int, float, np.ndarray, 'Tensor']) -> 'Tensor':
        """
        Implements slicing for the tensor using the [] operator.
        
        Parameters:
        - key (Union[int, slice, np.ndarray, Tuple[Union[int, slice, np.ndarray], ...]]): Key to slice the tensor
        - value (Union[int, float, np.ndarray, Tensor]): Value to set at the specified key
        """
        
        # Extract value data to avoid closure over Tensor object
        value_data = value.data if isinstance(value, Tensor) else value
        
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Set the value at the specified key (modifies in place)
            x_data[key] = value_data
            
            # Return the modified data
            return x_data, -1
            
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:
            # Use direct assignment which handles slices correctly
            np.add.at(out_buffer, key, out_grad)  # type: ignore
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    ########################
    ###### Properties ######
    ########################
    
    @property
    def shape(self) -> tuple:
        """
        Method to return the shape of the tensor
        
        Returns:
        - tuple: Shape of the tensor
        """
        
        # Return the shape of the tensor
        return self.data.shape


    ###########################
    ##### Public methods ######
    ###########################
    

    def zero_grad(self) -> None:
        """
        Method to reset the gradient of the tensor
        """
        
        # Reset the gradient of the tensor
        self.grad = None


    def backward(self, retain_graph: bool = False) -> None:
        """
        Method to backpropagate the gradient through the computation graph
        
        Parameters:
        - retain_graph (bool): Flag to indicate if the computation graph should be retained
        
        Raises:
        - ValueError: If the gradient is not computed for the tensor
        """
        
        # Propagate the gradient backward
        if self.grad is None:
            # Initialize the gradient with ones if not already computed
            self.grad = np.ones_like(self.data, dtype=self.data.dtype)
            
        # Build the topological order using iterative DFS
        visited: set[int] = set()
        topological_order: list['Tensor'] = []
        
        # Create a stack for iterative DFS
        stack: list[tuple['Tensor', bool]] = [(self, False)]

        # Perform DFS to build the topological order until all nodes are processed
        while stack:
            # Pop the next tensor to process
            tensor, processed = stack.pop()
            
            # Get the unique id of the tensor for visited tracking
            tensor_id = id(tensor)

            # Check if the tensor has been processed
            if processed:
                # Second visit: all children processed, add to order
                topological_order.append(tensor)

            # Check if the tensor is already visited
            elif tensor_id not in visited:
                # First visit: mark as visited and schedule children
                visited.add(tensor_id)
                
                # Push self again to be added after children
                stack.append((tensor, True))
                
                # Push children (will be processed before self)
                for child in tensor._prev:
                    if id(child) not in visited:
                        stack.append((child, False))
        
        # Backpropagate the gradient in the topological order (reverse order)
        for t in reversed(topological_order):
            # Backpropagate the gradient
            t._backward()
            
        # Clear the graph after backpropagation
        if not retain_graph:
            # Clear the graph to free up memory
            self.clear_graph()
      
    
    def copy(self) -> 'Tensor':
        """
        Method to create a copy of the tensor
        
        Returns:
        - Tensor: Copy of the tensor
        """
        
        # Create and return a copy of the tensor
        return Tensor(data=self.data.copy(), requires_grad=self.requires_grad, is_parameter=self.is_parameter)  
    
    
    def detach(self) -> 'Tensor':
        """
        Method to detach the tensor from the computation graph
        
        Returns:
        - Tensor: Detached tensor
        """
        
        # Create a new tensor with the same data but without gradient computation
        return Tensor(data=self.data, requires_grad=False, is_parameter=self.is_parameter)


    def clear_graph(self) -> None:
        """
        Method to clear the computation graph using iterative approach.
        Handles circular references and avoids creating new objects.
        """
        
        # Use iterative approach to avoid recursion overhead and stack limits
        visited: set[int] = set()
        stack: list['Tensor'] = [self]
        
        # Iterate until all nodes are processed
        while stack:
            # Pop the next tensor to process
            tensor = stack.pop()
            tensor_id = id(tensor)
            
            # Skip if already visited
            if tensor_id in visited:
                continue
                
            # Mark as visited
            visited.add(tensor_id)
            
            # Add children to stack before clearing
            for child in tensor._prev:
                if id(child) not in visited:
                    stack.append(child)
            
            # Clear current tensor's graph references
            tensor._backward = _noop
            tensor._prev.clear()
        
        # Clear the data tape once at the end
        tape_clear()


    def to_numpy(self) -> np.ndarray:
        """
        Method to convert the tensor to a numpy array
        
        Returns:
        - np.ndarray: Numpy array containing the data of the tensor
        """
        
        # Return the data of the tensor as a numpy array
        return self.data
    
    
    @classmethod
    def count_live(cls) -> int:
        """
        Method to count the number of live tensors in memory
        
        Returns:
        - int: Number of live tensors in memory
        """
        
        # Count the number of live tensors in memory
        return len(cls._live_tensors)
    
    
    ###########################
    ####### Activations #######
    ###########################
    
    def sigmoid(self) -> 'Tensor':
        """
        Method to compute the sigmoid of the tensor
        
        Returns:
        - Tensor: Tensor containing the sigmoid of the current tensor
        """
        
        # Extract input data size before defining closures
        input_data = self.data
        data_size = input_data.size
        
        # Define the forward function
        def forward(*args, **kwargs) -> tuple[np.ndarray, int]:
            # Create an empty array to store the output data
            out_data = np.empty_like(input_data)
            
            # Compute the sigmoid function
            sigmoid_forward(input_data.ravel(), out_data.ravel(), data_size)
            
            # Save the output data and size in the data tape and return it
            return out_data, tape_push((out_data, data_size))
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # Extract the output data from the saved data
            if saved_data is None:
                # If the output data is not found, raise an error
                raise ValueError("Output data not found in the data tape")
            
            # Extract the output data and size from the saved data
            out_data = saved_data[0]
            saved_size = saved_data[1]
            
            # Compute the gradient of the sigmoid function
            sigmoid_gradient(out_data.ravel(), out_grad.ravel(), out_buffer.ravel(), saved_size)
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    def silu(self) -> 'Tensor':
        """
        Method to compute the SiLU (Sigmoid Linear Unit) of the tensor
        
        Returns:
        - Tensor: Tensor containing the SiLU of the current tensor
        """
        
        # Compute and return the SiLU of the tensor
        return self * self.sigmoid()
    
    
    def relu(self) -> 'Tensor':
        """
        Method to compute the ReLU of the tensor
        
        Returns:
        - Tensor: Tensor containing the ReLU of the current tensor
        """
        
        # Extract input data size before defining closures
        input_data = self.data
        data_size = input_data.size
        
        # Define the forward function
        def forward(*args, **kwargs) -> tuple[np.ndarray, int]:
            # Create an empty array to store the output data
            out_data = np.empty_like(input_data)
            
            # Compute the ReLU function
            relu_forward(input_data.ravel(), out_data.ravel(), data_size)

            # Save the output data and size in the data tape and return it
            return out_data, tape_push((out_data, data_size))
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # Extract the output data from the saved data
            if saved_data is None:
                # If the output data is not found, raise an error
                raise ValueError("Output data not found in the data tape")
            
            # Extract the output data and size from the saved data
            out_data = saved_data[0]
            saved_size = saved_data[1]
            
            # Compute the gradient of the ReLU function
            relu_gradient(out_data.ravel(), out_grad.ravel(), out_buffer.ravel(), saved_size)
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    def tanh(self) -> 'Tensor':
        """
        Method to compute the hyperbolic tangent of the tensor
        
        Returns:
        - Tensor: Tensor containing the hyperbolic tangent of the current tensor
        """
        
        # Extract input data size before defining closures
        input_data = self.data
        data_size = input_data.size
        
        # Define the forward function
        def forward(*args, **kwargs) -> tuple[np.ndarray, int]:
            # Create an empty array to store the output data
            out_data = np.empty_like(input_data)
            
            # Compute the tanh function
            tanh_forward(input_data.ravel(), out_data.ravel(), data_size)
            
            # Save the output data and size in the data tape and return it
            return out_data, tape_push((out_data, data_size))
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # Extract the output data from the saved data
            if saved_data is None:
                # If the output data is not found, raise an error
                raise ValueError("Output data not found in the data tape")
            
            # Extract the output data and size from the saved data
            out_data = saved_data[0]
            saved_size = saved_data[1]
            
            # Compute the gradient of the tanh function
            tanh_gradient(out_data.ravel(), out_grad.ravel(), out_buffer.ravel(), saved_size)
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    def softmax(self, axis: int = -1) -> 'Tensor':
        """
        Method to compute the softmax of the tensor
        
        Parameters:
        - axis (int): Axis along which the softmax needs to be computed
        
        Returns:
        - Tensor: Tensor containing the softmax of the current tensor
        """
        
        # Extract input data before defining closures
        input_data = self.data
        input_ndim = input_data.ndim
        
        # Extract the number of elements in the input tensor
        k = input_data.shape[-1]
        n = input_data.size // k
        
        # Define the forward function
        def forward(*args, **kwargs) -> tuple[np.ndarray, int]:     
            # Compute the axis for softmax
            ax = axis % input_ndim
            
            # If the axis is not the last one, compute softmax along the specified axis
            if ax != input_ndim - 1:
                # Compute the maximum value along the specified axis
                out_data = np.exp(input_data) / np.sum(np.exp(input_data), axis=ax, keepdims=True)
            # If the axis is the last one, compute softmax using the kernel function
            else:
                # Create an empty array to store the output data
                out_data = np.empty_like(input_data)
            
                # Compute the softmax function
                softmax_forward(input_data.ravel(), out_data.ravel(), n, k)

            # Save the output data and dimensions in the data tape and return it
            return out_data, tape_push((out_data, n, k))
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # Extract the output data from the saved data
            if saved_data is None:
                # If the output data is not found, raise an error
                raise ValueError("Output data not found in the data tape")

            # Extract the output data and dimensions from the saved data
            out_data = saved_data[0]
            saved_n = saved_data[1]
            saved_k = saved_data[2]
            
            # Compute the gradient of the softmax function
            softmax_gradient(out_data.ravel(), out_grad.ravel(), out_buffer.ravel(), saved_n, saved_k)
            
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    def log_softmax(self, axis: int = -1) -> 'Tensor':
        """
        Method to compute the log softmax of the tensor
        
        Parameters:
        - axis (int): Axis along which the log softmax needs to be computed
        
        Returns:
        - Tensor: Tensor containing the log softmax of the current tensor
        """
        
        # Extract input data before defining closures
        input_data = self.data
        input_ndim = input_data.ndim
        
        # Extract the number of elements in the input tensor
        k = input_data.shape[-1]
        n = input_data.size // k
        
        # Define the forward function
        def forward(*args, **kwargs) -> tuple[np.ndarray, int]:
            # Compute the axis for log softmax
            ax = axis % input_ndim
            
            # If the axis is not the last one, compute log softmax along the specified axis
            if ax != input_ndim - 1:
                # Compute the maximum value along the specified axis
                m = np.max(input_data, axis=axis, keepdims=True)
                
                # Subtract the maximum value from the input data
                y = input_data - m
                
                # Compute the log sum of exponentials
                logsum = np.log(np.sum(np.exp(y), axis=axis, keepdims=True))
                
                # Compute the log softmax
                out_data = y - logsum
            
            # If the axis is the last one, compute log softmax using the kernel function
            else:
                # Create an empty array to store the output data
                out_data = np.empty_like(input_data)
                
                # Compute the log softmax function
                log_softmax_forward(input_data.ravel(), out_data.ravel(), n, k)

            # Save the output data and dimensions in the data tape and return it
            return out_data, tape_push((out_data, n, k))

        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # Extract the output data from the saved data
            if saved_data is None:
                # If the output data is not found, raise an error
                raise ValueError("Output data not found in the data tape")
            
            # Extract the output data and dimensions from the saved data
            out_data = saved_data[0]
            saved_n = saved_data[1]
            saved_k = saved_data[2]
            
            # Compute the gradient of the log softmax function
            log_softmax_gradient(out_data.ravel(), out_grad.ravel(), out_buffer.ravel(), saved_n, saved_k)
            
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    ###########################
    ######## Functions ########
    ###########################
    
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        Method to compute the sum of the tensor along the specified axis
        
        Parameters:
        - axis (int): Axis along which the sum needs to be computed
        - keepdims (bool): Flag to indicate if the dimensions need to be kept
        
        Returns:
        - Tensor: Tensor containing the sum of the current tensor along the specified axis
        """
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the sum of the tensor along the specified axis
            return sum_forward(x_data, axis=axis, keepdims=keepdims), -1

        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Backprop the gradient through the sum operation
            sum_backward(out_grad=out_grad, out_buffer=out_buffer, axis=axis, keepdims=keepdims)

        # Return the tensor operation withe the specified forward and backward functions         
        return tensor_unary_op(
            t = self, 
            forward_fn = forward, 
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        Method to compute the maximum value of the tensor along the specified axis
        
        Parameters:
        - axis (Optional[Union[int, Tuple[int, ...]]]): Axis along which to compute the maximum value
        - keepdims (bool): Whether to keep the dimensions of the input tensor
        
        Returns:
        - Tensor: Maximum value of the tensor along the specified axis
        """
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # If axis is None, compute the maximum value of the flattened tensor
            if axis is None:
                # Create a buffer to store the maximum value and its index
                buf = np.zeros((1,), dtype=x_data.dtype)
                idx = np.zeros((1,), dtype=np.int64)
                
                # Compute the maximum value of the flattened tensor
                max_flat_forward(x_data.ravel(), buf, idx)
                
                # If keepdims is True, create an output tensor with the same shape as the input tensor
                if keepdims:
                    out_data = np.full([1]*x_data.ndim, buf[0], dtype=x_data.dtype)
                else:
                    out_data = buf[0]
                    
                # Save the index in the data tape
                tape_idx = tape_push((idx, None))
                
            # If axis is not None, compute the maximum value along the specified axis
            else:
                # Compute the maximum value along the specified axis
                out_data = np.max(x_data, axis=axis, keepdims=keepdims)
                
                # Save x_data for the backward pass (needed for creating mask)
                tape_idx = tape_push((None, x_data.copy()))
                
            # Return the output data and tape index
            return out_data, tape_idx
                
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # If axis is None, compute the gradient for the flattened tensor
            if axis is None:
                if saved_data is None or saved_data[0] is None:
                    raise ValueError("Index not found in the data tape")
                
                # Compute the gradient of the max function for flattened tensor
                max_flat_gradient(saved_data[0], np.array([out_grad]).ravel(), out_buffer.ravel())
                
            # If axis is not None, compute the gradient along the specified axis
            else:
                # Retrieve x_data from saved_data
                if saved_data is None or saved_data[1] is None:
                    raise ValueError("Input data not found in the data tape")

                # Retrieve the expanded gradient
                x_data = saved_data[1]
                expanded = out_grad
                
                # Expand the gradient to match the shape of x_data if keepdims is False
                if not keepdims:
                    expanded = np.expand_dims(expanded, axis=axis)
                    
                # Create a mask to identify the maximum values
                mask = (x_data == expanded)
                count = np.sum(mask, axis=axis, keepdims=True)
                grad_x = mask * (expanded / count)
                
                # Accumulate gradient in out_buffer in-place
                np.add(out_buffer, grad_x, out=out_buffer)

        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
   
   
    def sqrt(self) -> 'Tensor':
        """
        Method to compute the square root of the tensor
        
        Returns:
        - Tensor: Tensor containing the square root of the current tensor
        """
        
        # Define the forward function - save input data for backward
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the square root and save input for backward pass
            return sqrt_forward(x_data), tape_push((x_data,))
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:
            # Retrieve x_data from saved_data
            x_data = saved_data[0] if saved_data else None
            
            # Compute the gradient of the square root function
            if x_data is not None:
                sqrt_backward(out_grad, out_buffer, x_data)
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self, 
            forward_fn = forward, 
            backward_fn = backward,
            tensor_cls = Tensor
        )
   
   
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        Method to compute the mean of the tensor along the specified axis
        
        Parameters:
        - axis (int): Axis along which to compute the mean
        - keepdims (bool): Whether to keep the dimensions of the input tensor
        
        Returns:
        - Tensor: Mean of the tensor along the specified axis
        """
        
        # Capture shape and size locally to avoid closure over entire tensor
        input_shape = self.data.shape
        input_size = self.data.size
        input_dtype = self.data.dtype
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the mean of the tensor along the specified axis
            return np.mean(x_data, axis=axis, keepdims=keepdims), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # If axis is None, compute the gradient for the flattened tensor
            if axis is None:
                # Compute the gradient of the mean function for flattened tensor
                buf = np.zeros((1,), dtype=input_dtype)
                buf[0] = out_grad
                inv = 1.0 / input_size
                
                # Compute the mean flat backward
                mean_flat_backward(buf, out_buffer.ravel(), inv)
                
            # If axis is not None, compute the gradient along the specified axis
            else:
                # Retrieve the expanded gradient
                grad = out_grad
                
                # Expand the gradient to match the shape of input tensor
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)

                # Compute the number of elements along the specified axis/axes
                if isinstance(axis, int):
                    num_elements_along_axis = input_shape[axis]
                elif isinstance(axis, tuple):
                    num_elements_along_axis = int(np.prod([input_shape[ax] for ax in axis]))
                else:
                    raise TypeError("axis must be an int or a tuple of ints")

                # Accumulate gradient in out_buffer in-place
                np.add(out_buffer, grad / num_elements_along_axis, out=out_buffer)

        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    def var(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, ddof: int = 1) -> 'Tensor':
        """
        Method to compute the variance of the tensor along the specified axis
        
        Parameters:
        - axis (int): Axis along which to compute the variance
        - keepdims (bool): Whether to keep the dimensions of the input tensor
        - ddof (int): Delta degrees of freedom for the variance computation
        
        Returns:
        - Tensor: variance of the tensor along the specified axis
        """
        
        # Compute the mean with keepdims=True to allow proper broadcasting.
        m = self.mean(axis=axis, keepdims=True)
        
        # Compute the squared difference between the tensor and the mean.
        diff = self - m
        
        # Compute the squared difference and take the mean along the specified axis.
        square_diff = diff * diff
        
        # Compute the variance by taking the mean of the squared difference.
        var = square_diff.mean(axis=axis, keepdims=keepdims)
        
        # Determine the number of elements over which the variance is computed.
        if axis is None:
            # If axis is None, the variance is computed over all elements.
            num_elements = self.data.size
            
        # If axis is an integer, the variance is computed over the elements in that axis.
        elif isinstance(axis, int):
            # If axis is an integer, the variance is computed over the elements in that axis.
            num_elements = self.data.shape[axis]
            
        # If axis is a tuple, the variance is computed over the elements in the specified axes.
        elif isinstance(axis, tuple):
            # If axis is a tuple, the variance is computed over the elements in the specified axes.
            num_elements = int(np.prod([self.data.shape[ax] for ax in axis]))
        else:
            # If axis is not an integer or a tuple, raise a TypeError.
            raise ValueError("axis must be an integer, a tuple of integers, or None")
        
        # If num_elements > ddof, apply the bessel correction to the variance.
        if ddof != 0 and num_elements > ddof:
            # Convert the population variance to the sample variance.
            var = var * (num_elements / (num_elements - 1))
        
        # Return the variance tensor
        return var
   
    
    def exp(self) -> 'Tensor':
        """
        Method to compute the exponential of the tensor
        
        Returns:
        - Tensor: Tensor containing the exponential of the current tensor
        """
        
        # Define the forward function - save input data for backward
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the exponential of the tensor and save input for backward
            return np.exp(x_data), tape_push((x_data,))
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:
            # Retrieve x_data from saved_data
            x_data = saved_data[0] if saved_data else None
            
            # Compute the gradient of the exponential function
            if x_data is not None:
                exp_gradient(out_grad.ravel(), x_data.ravel(), out_buffer.ravel())
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )


    def log(self) -> 'Tensor':
        """
        Method to compute the natural logarithm of the tensor
        
        Returns:
        - Tensor: Tensor containing the natural logarithm of the current tensor
        """
        
        # Define the forward function - save input data for backward
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the natural logarithm of the tensor and save input for backward
            return np.log(x_data), tape_push((x_data,))
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:
            # Retrieve x_data from saved_data
            x_data = saved_data[0] if saved_data else None
            
            # Compute the gradient of the natural logarithm function
            if x_data is not None:
                log_gradient(x_data.ravel(), out_grad.ravel(), out_buffer.ravel())
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )

     
    def transpose(self, axes: tuple) -> 'Tensor':
        """
        Method to compute the transpose of the tensor
        
        Parameters:
        - axes (tuple): Permutation of the dimensions
        
        Returns:
        - Tensor: Transposed tensor
        """
        
        # Precompute inverse axes to avoid closure over mutable state
        inv_axes = tuple(np.argsort(axes))
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the transpose of the tensor
            return x_data.transpose(axes), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:
            # Transpose the gradient to match the original tensor using pre-computed inv_axes
            grad_x = out_grad.transpose(inv_axes)
            
            # Accumulate the gradient in the output buffer in-place
            np.add(out_buffer, grad_x, out=out_buffer)
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )


    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> 'Tensor':
        """
        Flattens a contiguous range of dimensions in the tensor.
        
        Parameters:
        - start_dim (int): First dimension to flatten (default: 0)
        - end_dim (int): Last dimension to flatten (default: -1, meaning last dimension)
        
        Returns:
        - Tensor: Flattened tensor
        
        Examples:
        - tensor.flatten() flattens all dimensions
        - tensor.flatten(1) flattens from dim 1 to the end (typical for batched data)
        - tensor.flatten(1, 2) flattens only dims 1 and 2
        """
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Perform the flatten operation
            out_data, original_shape = flatten_forward(x_data, start_dim, end_dim)
            
            # Save the original shape in the data tape for the backward pass and return the output data
            return out_data, tape_push((original_shape,))
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # Extract the original shape from the saved data
            if saved_data is None:
                raise ValueError("Original shape not found in the data tape")
            
            # Get the original shape
            original_shape = saved_data[0]
            
            # Compute the gradient by reshaping back to original shape
            flatten_backward(out_grad, out_buffer, original_shape)
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )


    def top_k(self, k: int, axis: int = -1, largest: bool = True, sorted: bool = True) -> Tuple['Tensor', 'Tensor']:
        """
        Method to compute the top k elements of the tensor along the specified axis
        
        Parameters:
        - k (int): Number of top elements to select
        - axis (int): Axis along which to select the top k elements
        - largest (bool): Whether to select the largest or smallest elements
        - sorted (bool): Whether to sort the selected elements
        
        Returns:
        - Tensor: Tensor containing the top k elements along the specified axis
        """
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], int]:
            # Compute the top k values and their indices using numpy
            values, indices = top_k_forward(x_data, k=k, dim=axis, largest=largest, sorted=sorted)

            # Push the indices to the data tape for use in the backward pass and return the values and indices
            return (values, indices), tape_push((indices,))

        # Define the backward function for values
        def backward_values(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # Extract the indices from the saved data
            if saved_data is None:
                raise ValueError("Indices not found in the data tape")

            # Get the indices
            indices = saved_data[0]

            # Compute the gradient by scattering it to the original positions
            top_k_backward(out_grad, out_buffer, indices, dim=axis)
            
        # Define the backward function for indices (no gradient)
        def backward_indices(*args, **kwargs) -> None:
            # No gradient is computed for indices
            pass

        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op_binary_output(
            t = self,
            forward_fn = forward,
            backward_fn_a = backward_values,
            backward_fn_b = backward_indices,
            tensor_cls = Tensor
        )


    def gather(self, axis: int, index: 'Tensor') -> 'Tensor':
        """
        Method to gather values from the tensor at specified indices along a given axis
        
        Parameters:
        - axis (int): Axis along which to gather the values
        - index (Tensor): Indices to gather the values from
        
        Returns:
        - Tensor: Tensor containing the gathered values
        """
        
        # Ensure index is integer type
        index_data = index.data.astype(np.int64)

        # Define the forward function
        def forward(x_data: np.ndarray, *args, **kwargs) -> tuple[np.ndarray, int]:
            # Perform the gather operation
            out_data = gather_forward(x_data, axis, index_data)

            # Return the gathered tensor
            return out_data, -1
            
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the gather operation
            gather_backward(out_grad, out_buffer, axis, index_data)

        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )


    def masked_fill(self, mask: Union[np.ndarray, 'Tensor'], value: float) -> 'Tensor':
        """
        Method to fill the tensor with a value where the mask is True
        
        Parameters:
        - mask (Union[np.ndarray, Tensor]): Mask to fill the tensor
        - value (float): Value to fill the tensor
        
        Returns:
        - Tensor: Tensor with the masked fill operation
        """
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:        
            # Extract the mask data
            mask_arr = mask if isinstance(mask, np.ndarray) else mask.data
            
            # Flatten the mask and input tensor data
            mask_flat = mask_arr.ravel()
            flat_data = x_data.ravel()
            
            # Prepare the output tensor and flatten it
            out_data = np.empty_like(x_data)
            out_flat = out_data.ravel()
            
            if isinstance(value, float) and not np.isfinite(value):
                # The value is negative infinity
                if value < 0:
                    # Perform the masked fill operation with negative infinity
                    masked_fill_forward_neg_inf(flat_data, mask_flat, out_flat)
                    
                # The value is positive infinity
                else:
                    # Perform the masked fill operation with positive infinity
                    masked_fill_forward_inf(flat_data, mask_flat, out_flat)
            else:
                # Cast the value to the appropriate type    
                fill_val = self.data.dtype.type(value)
                
                # Perform the masked fill operation
                masked_fill_forward(flat_data, mask_flat, fill_val, out_flat)
        
            # Save the mask in the data tape to use it in the backward pass and return the output data
            return out_data, tape_push((mask_flat,))
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # Extract the mask from the saved data
            if saved_data is None:
                # If the mask is not found, raise an error
                raise ValueError("Mask not found in the data tape")
            
            # Backpropagate the gradient through the masked fill operation
            masked_fill_gradient(saved_data[0], out_grad.ravel(), out_buffer.ravel())
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )


    def scatter(self, axis: int, index: 'Tensor', src: 'Tensor', reduction: Optional[str] = None) -> 'Tensor':
        """
        Method to scatter values from the source tensor into the current tensor at specified indices
        
        Parameters:
        - axis (int): Axis along which to scatter the values
        - index (Tensor): Indices where the values need to be scattered
        - src (Tensor): Source values to scatter
        - reduction (Optional[str]): Optional reduction method to apply ('add', 'multiply')
        
        Returns:
        - Tensor: Tensor with the scattered values
        """

        # Define the forward function
        def forward(x_data: np.ndarray, *args, **kwargs) -> tuple[np.ndarray, int]:
            # Perform the scatter operation and return the scattered tensor
            return scatter_forward(x_data, axis, index.data, src.data, reduction), -1
            
        # Define the backward function for tensor x
        def backward_x(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the multiplication operation
            scatter_backward_x(out_grad, out_buffer, axis, index.data, reduction)

        # Define the backward function for tensor src
        def backward_src(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the multiplication operation
            scatter_backward_src(out_grad, out_buffer, axis, index.data, reduction)

        # Return the tensor operation with the specified forward and backward functions
        return tensor_binary_op(
            t1 = self,
            t2 = src,
            forward_fn = forward,
            backward_fn_a = backward_x,
            backward_fn_b = backward_src,
            tensor_cls = Tensor
        )


    def clip(self, min_val: float, max_val: float) -> 'Tensor':
        """
        Method to clip the tensor within the specified range
        
        Parameters:
        - min_val (float): Minimum value to clip
        - max_val (float): Maximum value to clip
        
        Returns:
        - Tensor: Clipped tensor
        """
        
        # Define the forward function - save input data for backward
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Create an output tensor with the same shape as the input tensor
            out_data = np.empty_like(x_data)
        
            # Clip the values of the tensor to the specified range
            clip_forward(x_data.ravel(), min_val, max_val, out_data.ravel())
            
            # Save input data for backward pass and return the output data
            return out_data, tape_push((x_data,))
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:
            # Retrieve x_data from saved_data
            x_data = saved_data[0] if saved_data else None
            
            # Compute the gradient of the clip function
            if x_data is not None:
                clip_gradient(x_data.ravel(), out_grad.ravel(), out_buffer.ravel(), min_val, max_val)
                
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    def squeeze(self, axis: int) -> 'Tensor':
        """
        Method to squeeze the tensor along the specified axis
        
        Parameters:
        - axis (int): Axis along which to squeeze the tensor
        
        Returns:
        - Tensor: Sequeezed tensor
        """
        
        # Capture original shape locally to avoid closure over entire tensor
        original_shape = self.data.shape
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Squeeze the tensor along the specified axis
            return np.squeeze(x_data, axis=axis), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:        
            # Unsqueeze the gradient along the same axis to match the original shape
            if axis is None:
                # For None case, we need to restore all squeezed dims
                grad_squeezed = out_grad
                for dim in sorted([i for i, size in enumerate(original_shape) if size == 1], reverse=True):
                    grad_squeezed = np.expand_dims(grad_squeezed, axis=dim)
            else:
                # For specific axis case
                grad_squeezed = np.expand_dims(out_grad, axis=axis)
            
            # Accumulate gradient in out_buffer in-place
            np.add(out_buffer, grad_squeezed, out=out_buffer)

        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    def unsqueeze(self, axis: int) -> 'Tensor':
        """
        Method to unsqueeze the tensor along the specified axis
        
        Parameters:
        - axis (int): Axis along which to unsqueeze the tensor
        
        Returns:
        - Tensor: Unsqueezed tensor
        """
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Unsqueeze the tensor along the specified axis
            return np.expand_dims(x_data, axis=axis), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:        
            # Compute the gradient of the unsqueeze operation
            unsqueeze_gradient(out_grad.ravel(), out_buffer.ravel())
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    def reshape(self, shape: tuple) -> 'Tensor':
        """
        Method to reshape the tensor
        
        Parameters:
        - shape (tuple): New shape of the tensor
        
        Returns:
        - Tensor: Reshaped tensor
        """
        
        # Capture original shape locally to avoid closure over entire tensor
        original_shape = self.data.shape
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Reshape the tensor to the specified new shape
            return x_data.reshape(shape), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:        
            # Reshape the gradient to match the original tensor shape
            grad_back = out_grad.reshape(original_shape)
            
            # Accumulate gradient in out_buffer in-place
            np.add(out_buffer, grad_back, out=out_buffer)
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    def repeat(self, repeats: int, axis: Optional[int] = None) -> 'Tensor':
        """
        Method to repeat the tensor along the specified dimensions
        
        Parameters:
        - repeats (Union[int, Tuple[int, ...]]): Number of times to repeat the tensor
        - axis (Union[int, Tuple[int, ...]]): Axis along which to repeat the tensor
        
        Returns:
        - Tensor: Repeated tensor
        """
        
        # Capture shape, size and dtype locally to avoid closure over entire tensor
        original_shape = self.data.shape
        original_size = self.data.size
        original_dtype = self.data.dtype
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # If the axis is None, flatten the tensor and repeat
            if axis is None:
                out_data = np.empty(original_size * repeats, dtype=original_dtype)
                repeat_forward(x_data.ravel(), repeats, out_data)
            else:
                out_data = np.repeat(x_data, repeats, axis=axis)
                
            # Return the repeated tensors
            return out_data, -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # If the axis is None, compute the gradient for the flattened tensor
            if axis is None:
                repeat_gradient(out_grad.ravel(), repeats, out_buffer.ravel())
            else:
                # Reduce the gradient along the specified axis
                grad_unrepeated = np.add.reduce(
                    out_grad.reshape(
                        *(original_shape[:axis]), original_shape[axis], repeats, *original_shape[axis+1:]
                    ), axis = axis+1
                )
                
                # Accumulate gradient in out_buffer in-place
                np.add(out_buffer, grad_unrepeated, out=out_buffer)

        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    def expand(self, *sizes: int) -> 'Tensor':
        """
        Expands the tensor to a larger size. 
        
        Passing -1 as the size for a dimension means not changing the size of that dimension.
        The tensor can only be expanded to a larger size along dimensions of size 1.
        Expanding a tensor does not allocate new memory, but creates a new view on the existing tensor.
        
        Parameters:
        - *sizes (int): The desired expanded size for each dimension
        
        Returns:
        - Tensor: Expanded tensor
        
        Examples:
        - tensor.expand(3, 4) expands a (1, 4) tensor to (3, 4)
        - tensor.expand(-1, -1, 4) expands a (2, 3, 1) tensor to (2, 3, 4)
        """
        
        # Convert sizes to tuple
        target_shape = tuple(sizes)
        original_shape = self.data.shape
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Expand the tensor to the target shape
            return expand_forward(x_data, target_shape), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient by reducing over expanded dimensions
            expand_backward(out_grad, out_buffer, original_shape)
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    def pad(self, pad_width: tuple) -> 'Tensor':
        """
        Method to pad the tensor along the specified dimensions
        
        Parameters:
        - pad_width (tuple): Width of the padding
        
        Returns:
        - Tensor: Padded tensor
        """
        
        # Capture shape and dtype locally to avoid closure over entire tensor
        input_shape = self.data.shape
        input_dtype = self.data.dtype
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Extract the padding widths for each dimension
            (_, _), (pt, pb), (pl, pr), (_, _) = pad_width
            batch_size, height, width, channels = input_shape
            
            # Create the output tensor with the new shape
            out_data = np.empty((batch_size, height + pt + pb, width + pl + pr, channels), dtype=input_dtype)
            
            # Perform the padding operation
            pad_forward(x_data, pt, pb, pl, pr, out_data)
            
            return out_data, -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:        
            # Extract the padding widths for each dimension
            (_, _), (pt, pb), (pl, pr), (_, _) = pad_width
            
            # Compute the gradient of the padding operation
            pad_gradient(out_grad, pt, pb, pl, pr, out_buffer)
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    def conv_2d(self, kernel: 'Tensor', stride: Tuple[int,int] = (1,1)) -> 'Tensor':
        """
        Method to compute the 2D convolution of the tensor
        
        Parameters:
        - kernel (Tensor): Kernel for the convolution
        - stride (Tuple[int,int]): Stride for the convolution. Default is (1,1)
        
        Returns:
        - Tensor: Tensor containing the 2D convolution of the tensor
        """
        
        # Define the forward function - save both inputs for backward
        def forward(x_data: np.ndarray, kernel_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Perform the convolution operation and save inputs for backward
            return conv_2d_forward(x_data, kernel_data, stride), tape_push((x_data, kernel_data))
        
        # Define the backward function for the input tensor
        def backward_x(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:
            # Retrieve kernel_data from saved_data
            _, kernel_data = saved_data if saved_data else (None, None)
            
            # Compute the gradient of the convolution with respect to the input tensor
            if kernel_data is not None:
                conv_2d_backward_x(out_grad=out_grad, out_buffer=out_buffer, kernel_data=kernel_data, stride=stride)
            
        # Define the backward function for the kernel
        def backward_w(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Any, *args, **kwargs) -> None:
            # Retrieve x_data from saved_data
            x_data, _ = saved_data if saved_data else (None, None)
            
            # Compute the gradient of the convolution with respect to the kernel
            if x_data is not None:
                conv_2d_backward_w(out_grad=out_grad, out_buffer=out_buffer, x_data=x_data, stride=stride)
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_binary_op(
            t1 = self,
            t2 = kernel, 
            forward_fn = forward,
            backward_fn_a = backward_x,
            backward_fn_b = backward_w,
            tensor_cls = Tensor
        )
    
    
    def max_pool_2d(self, kernel_size: Tuple[int,int] = (2,2), stride: Tuple[int,int] = (2,2)) -> 'Tensor':
        """
        Method to compute the 2D max pooling of the tensor
        
        Parameters:
        - kernel_size (Tuple[int,int]): Kernel size for the pooling. Default is (2,2)
        - stride (Tuple[int,int]): Stride for the pooling. Default is (2,2)
        
        Returns:
        - Tensor: Tensor containing the 2D max pooling of the tensor
        """
        
        # Define the stride values
        stride_height, stride_width = stride
        kernel_height, kernel_width = kernel_size
        
        # Capture dtype locally
        input_dtype = self.data.dtype
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:        
            # Extract the input dimensions
            batch_size, height, width, channels = x_data.shape
            
            # Compute the output dimensions
            out_height = (height - kernel_height) // stride_height + 1
            out_width = (width - kernel_width) // stride_width + 1
            
            # Check if the kernel or stride is too large for the input size
            if out_height < 1 or out_width < 1:
                raise ValueError("Kernel size or stride too large for input size.")

            # Create the output array
            out_data = np.empty((batch_size, out_height, out_width, channels), dtype=input_dtype)
            
            # Initialize the indices for max pooling
            arg_i = np.zeros_like(out_data, dtype=np.int32)
            arg_j = np.zeros_like(out_data, dtype=np.int32)

            # Perform the max pooling operation
            max_pool_2d_forward(x_data, kernel_height, kernel_width, stride_height, stride_width, out_data, arg_i, arg_j)

            # Save the indices in the data tape
            return out_data, tape_push((arg_i, arg_j))
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # Extract the indices from the saved data
            if saved_data is None:
                # If the indices are not found, raise an error
                raise ValueError("Indices not found in the data tape")
            
            # Unpack the saved data
            arg_i, arg_j = saved_data
            
            # Backprop the gradient through the max pooling operation
            max_pool_2d_gradient(arg_i, arg_j, out_grad, stride_height, stride_width, out_buffer)
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    @staticmethod
    def concat(tensors: List['Tensor'], axis: int = 0) -> 'Tensor':
        """
        Concatenate a list of tensors along the specified axis.
        
        Parameters:
        - tensors (List[Tensor]): List of input tensors
        - axis (int): Axis along which to concatenate the tensors
        
        Returns:
        - Tensor: Concatenated tensor
        """

        # Define the forward function
        def forward(tensors_list: List[np.ndarray]) -> tuple[np.ndarray, int]:
            # Perform the concatenation operation
            out, offsets = concat_forward(tensors_list, axis)
                
            # Save the offsets in the data tape to use it in the backward pass and return the output data
            return out, tape_push((offsets,))
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, tensor_idx: int, saved_data: Optional[Tuple[Any, ...]]) -> None:
            # Check if the offsets are valid
            if saved_data is None:
                raise ValueError("Offsets not found in the data tape")
            
            # Call the kernel function to compute the gradient of the concatenation operation
            concat_backward(out_grad, out_buffer, saved_data[0], tensor_idx)

        # Return the tensor operation with the specified forward and backward functions
        return tensor_nary_op(
            tensors = tensors, 
            forward_fn = forward, 
            backward_fn = backward,
            tensor_cls = Tensor
        )
        
    
    @staticmethod
    def stack(tensors: List['Tensor'], axis: int = 0) -> 'Tensor':
        """
        Stack a list of tensors along the specified axis.
        
        Parameters:
        - tensors (List[Tensor]): List of input tensors
        - axis (int): Axis along which to stack the tensors
        
        Returns:
        - Tensor: Stacked tensor
        """
        
        # Define the forward function
        def forward(tensors_list: List[np.ndarray]) -> tuple[np.ndarray, int]:
            # Perform the stacking operation and return the stacked tensor
            return stack_forward(tensors_list, axis), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, tensor_idx: int, *args, **kwargs) -> None:
            # Call the kernel function to compute the gradient of the stacking operation
            stack_backward(out_grad, out_buffer, axis, tensor_idx)

        # Return the tensor operation with the specified forward and backward functions
        return tensor_nary_op(
            tensors = tensors,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
       
     
    @staticmethod
    def split(tensor: 'Tensor', indices_or_sections: Union[int, List[int]], axis: int = 0) -> List['Tensor']:
        """
        Split a tensor into multiple sub-tensors along the specified axis.
        
        Parameters:
        - tensor (Tensor): Input tensor to be split
        - indices_or_sections (Union[int, List[int]]): Number of sections or list of indices to split at
        - axis (int): Axis along which to split the tensor
        
        Returns:
        - List[Tensor]: List of sub-tensors resulting from the split operation
        """
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[List[np.ndarray], int]:
            # Perform the split operation and return the list of sub-tensors
            return split_forward(x_data, indices_or_sections, axis), -1
        
        # Define the backward function
        def backward(out_grads: List[np.ndarray], out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Call the kernel function to compute the gradient of the split operation
            split_backward(out_grads, out_buffer, axis)

        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op_multiple_outputs(
            t = tensor,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
        
        
    @staticmethod
    def einsum(subscripts: str, *operands: 'Tensor') -> 'Tensor':
        """
        Perform Einstein summation on the given tensors.
        
        Parameters:
        - subscripts (str): Einsum subscript string (e.g., 'ij,jk->ik' for matrix multiplication)
        - operands (Tensor): Input tensors
        
        Returns:
        - Tensor: Result of the einsum operation
        
        Examples:
        - Matrix multiplication: Tensor.einsum('ij,jk->ik', A, B)
        - Batch matrix multiplication: Tensor.einsum('bij,bjk->bik', A, B)
        - Transpose: Tensor.einsum('ij->ji', A)
        - Trace: Tensor.einsum('ii->', A)
        - Outer product: Tensor.einsum('i,j->ij', a, b)
        - Attention scores: Tensor.einsum('bhqd,bhkd->bhqk', Q, K)
        """
        
        # Convert operands to list for tensor_nary_op
        tensors_list = list(operands)
        
        # Store operand data for backward pass
        operands_data = tuple(t.data for t in operands)

        # Define the forward function
        def forward(tensors_data: List[np.ndarray]) -> tuple[np.ndarray, int]:
            # Perform the einsum operation and return the result
            return einsum_forward(subscripts, *tensors_data), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, tensor_idx: int, *args, **kwargs) -> None:
            # Compute the gradient for the specified operand
            grad = einsum_backward(subscripts, out_grad, operands_data, tensor_idx)
            
            # Accumulate the gradient in-place
            np.add(out_buffer, grad, out=out_buffer)

        # Return the tensor operation with the specified forward and backward functions
        return tensor_nary_op(
            tensors = tensors_list,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )