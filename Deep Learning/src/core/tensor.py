import weakref
import numpy as np
from typing import Optional, Union, Tuple, List, Callable, Any

from .functional.kernel import *
from .functional.tape import tape_push
from .utils.context_manager import _NO_GRAD
from .functional.base import tensor_unary_op, tensor_binary_op, tensor_nary_op, accumulate_gradient



class Tensor:
    
    ##########################
    ##### Class variables ####
    ##########################
    
    # Class variable to keep track of live tensors
    _live_tensors = weakref.WeakSet()
    
    
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
        
        # Check if the data is a numpy array
        if isinstance(data, np.ndarray):
            # Check if the data is already of the correct dtype
            if data.dtype != dtype:
                # Convert the data to the correct dtype
                self.data = data.astype(dtype, copy=False)
            else:
                # Assign the data directly if already of the correct dtype
                self.data = data
        else:
            # Convert the data to a numpy array of the correct dtype
            self.data = np.array(data, dtype=dtype)
        
        # Gradient tracking flags
        self.is_parameter = is_parameter
        self.requires_grad = False if _NO_GRAD else requires_grad
        
        # Initialize gradient and graph metadata
        self.grad = None
        self._backward: Callable = lambda: None
        self._prev: set['Tensor'] = set()

        # Register live tensor
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
            
        # Define the forward function
        def forward(a_data: np.ndarray, b_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the product of the two tensors
            return add_forward(a_data, b_data), -1
        
        # Define the backward function for tensor a
        def backward_a(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the multiplication operation
            add_backward(out_grad=out_grad, out_buffer=out_buffer, target_shape=self.data.shape)
            
        # Define the backward function for tensor b
        def backward_b(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the multiplication operation
            add_backward(out_grad=out_grad, out_buffer=out_buffer, target_shape=other.data.shape)
                
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
            
        # Define the forward function
        def forward(a_data: np.ndarray, b_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the product of the two tensors
            return sub_forward(a_data, b_data), -1
        
        # Define the backward function for tensor a
        def backward_a(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the multiplication operation
            sub_backward_a(out_grad=out_grad, out_buffer=out_buffer, target_shape=self.data.shape)
            
        # Define the backward function for tensor b
        def backward_b(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the multiplication operation
            sub_backward_b(out_grad=out_grad, out_buffer=out_buffer, target_shape=other.data.shape)
                
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
            
        # Define the forward function
        def forward(a_data: np.ndarray, b_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the product of the two tensors
            return mul_forward(a_data, b_data), -1
        
        # Define the backward function for tensor a
        def backward_a(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the multiplication operation
            mul_backward_a(out_grad=out_grad, out_buffer=out_buffer, target_shape=self.data.shape, b_data=other.data)
            
        # Define the backward function for tensor b
        def backward_b(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the multiplication operation
            mul_backward_b(out_grad=out_grad, out_buffer=out_buffer, target_shape=other.data.shape, a_data=self.data)
                
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
            
        # Define the forward function
        def forward(a_data: np.ndarray, b_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the product of the two tensors
            return div_forward(a_data, b_data), -1
        
        # Define the backward function for tensor a
        def backward_a(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the multiplication operation
            div_backward_a(out_grad=out_grad, out_buffer=out_buffer, target_shape=self.data.shape, b_data=other.data)
            
        # Define the backward function for tensor b
        def backward_b(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the multiplication operation
            div_backward_b(out_grad=out_grad, out_buffer=out_buffer, a_data=self.data, b_data=other.data)
                
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
            
        # Define the forward function
        def forward(a_data: np.ndarray, b_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the product of the two tensors
            return matmul_forward(a_data, b_data), -1
        
        # Define the backward function for tensor a
        def backward_a(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the multiplication operation
            matmul_backward_a(out_grad=out_grad, out_buffer=out_buffer, b_data=other.data)
            
        # Define the backward function for tensor b
        def backward_b(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the multiplication operation
            matmul_backward_b(out_grad=out_grad, out_buffer=out_buffer, a_data=self.data)
                
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
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the power of the tensor
            return pow_forward(self.data, power), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, *args, **kwargs) -> None:            
            # Compute the gradient of the power function
            grad = pow_gradient(out_grad, self.data, power)
            
            # Accumulate the gradient into self.grad.
            accumulate_gradient(self, grad)
            
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
    
    
    def __getitem__(self, key: Union[int, slice, np.ndarray, Tuple[Union[int, slice, np.ndarray], ...]]) -> 'Tensor':
        """
        Implements slicing for the tensor using the [] operator.
        
        Parameters:
        - key (Union[int, slice, np.ndarray, Tuple[Union[int, slice, np.ndarray], ...]]): Key to slice the tensor

        Returns:
        - Tensor: A new tensor with data = self.data[key].
        """
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Return the sliced data
            return self.data[key], -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the slice operation
            grad = np.zeros_like(self.data)
            
            # Use direct assignment which handles slices correctly
            np.add.at(grad, key, out_grad) # type: ignore
                    
            # Accumulate the gradient into self.grad.
            accumulate_gradient(self, grad)
            
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    

    ###########################
    ##### Public methods ######
    ###########################
    
    def shape(self) -> tuple:
        """
        Method to return the shape of the tensor
        
        Returns:
        - tuple: Shape of the tensor
        """
        
        # Return the shape of the tensor
        return self.data.shape
    

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
            
        # Build the topological order of the tensors
        visited = set()
        
        # Function to build the topological order
        topological_order: list['Tensor'] = []
        def build_topo(t: 'Tensor') -> None:
            # If the tensor is not visited, add it to the visited set and build the topological order
            if t not in visited:
                # Add the tensor to the visited set
                visited.add(t)
                
                # Recursively build the topological order for the children
                for child in t._prev:
                    # Build the topological order for the child
                    build_topo(child)
                
                # Append the tensor to the topological order
                topological_order.append(t)
                
        # Build the topological order
        build_topo(self)
        
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
    
    
    def clear_graph(self, visited: Optional[set] = None) -> None:
        """
        Method to clear the computation graph
        
        Parameters:
        - t (Tensor): Tensor to clear the graph
        - visited (set): Set to store the visited tensors
        """
        
        # Initialize the visited set
        if visited is None:
            visited = set()
            
        # Check if the tensor is not visited
        if self not in visited:
            # Add the tensor to the visited set
            visited.add(self)
            
        # Iterate over the children
        for child in list(self._prev):
            # Clear the graph for the child
            child.clear_graph(visited)
            
        # Clear the graph for the current tensor and the backward function
        self._backward = lambda: None
        self._prev.clear()
    
    
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
        
        # Define the forward function
        def forward(*args, **kwargs) -> tuple[np.ndarray, int]:
            # Create an empty array to store the output data
            out_data = np.empty_like(self.data)
            
            # Compute the sigmoid function
            sigmoid_forward(self.data.ravel(), out_data.ravel(), self.data.size)
            
            # Save the output data in the data tape
            tape_idx = tape_push((out_data,))
            
            # Return the output data
            return out_data, tape_idx
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # Extract the output data from the saved data
            if saved_data is None:
                # If the output data is not found, raise an error
                raise ValueError("Output data not found in the data tape")
            
            # Extract the output data from the saved data
            out_data = saved_data[0]
            
            # Compute the gradient of the sigmoid function
            sigmoid_gradient(out_data.ravel(), out_grad.ravel(), out_buffer.ravel(), self.data.size)
        
        # Return the tensor operation with the specified forward and backward functions
        return tensor_unary_op(
            t = self,
            forward_fn = forward,
            backward_fn = backward,
            tensor_cls = Tensor
        )
    
    
    def relu(self) -> 'Tensor':
        """
        Method to compute the ReLU of the tensor
        
        Returns:
        - Tensor: Tensor containing the ReLU of the current tensor
        """
        
        # Define the forward function
        def forward(*args, **kwargs) -> tuple[np.ndarray, int]:
            # Create an empty array to store the output data
            out_data = np.empty_like(self.data)
            
            # Compute the sigmoid function
            relu_forward(self.data.ravel(), out_data.ravel(), self.data.size)
            
            # Save the output data in the data tape
            tape_idx = tape_push((out_data,))
            
            # Return the output data
            return out_data, tape_idx
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # Extract the output data from the saved data
            if saved_data is None:
                # If the output data is not found, raise an error
                raise ValueError("Output data not found in the data tape")
            
            # Extract the output data from the saved data
            out_data = saved_data[0]
            
            # Compute the gradient of the sigmoid function
            relu_gradient(out_data.ravel(), out_grad.ravel(), out_buffer.ravel(), self.data.size)
        
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
        
        # Define the forward function
        def forward(*args, **kwargs) -> tuple[np.ndarray, int]:
            # Create an empty array to store the output data
            out_data = np.empty_like(self.data)
            
            # Compute the sigmoid function
            tanh_forward(self.data.ravel(), out_data.ravel(), self.data.size)
            
            # Save the output data in the data tape
            tape_idx = tape_push((out_data,))
            
            # Return the output data
            return out_data, tape_idx
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # Extract the output data from the saved data
            if saved_data is None:
                # If the output data is not found, raise an error
                raise ValueError("Output data not found in the data tape")
            
            # Extract the output data from the saved data
            out_data = saved_data[0]
            
            # Compute the gradient of the sigmoid function
            tanh_gradient(out_data.ravel(), out_grad.ravel(), out_buffer.ravel(), self.data.size)
        
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
        
        # Extract the number of elements in the input tensor and create an empty array to store the output data
        k = self.data.shape[-1]
        n = self.data.size // k
        
        # Define the forward function
        def forward(*args, **kwargs) -> tuple[np.ndarray, int]:     
            # Extract number of elements in the input tensor
            ndim = self.data.ndim
            
            # Compute the axis for softmax
            ax = axis % ndim
            
            # If the axis is not the last one, compute softmax along the specified axis
            if ax != ndim - 1:
                # Compute the maximum value along the specified axis
                out_data = np.exp(self.data) / np.sum(np.exp(self.data), axis=ax, keepdims=True)
            # If the axis is the last one, compute softmax using the kernel function
            else:
                # Create an empty array to store the output data
                out_data = np.empty_like(self.data)
            
                # Compute the softmax function
                softmax_forward(self.data.ravel(), out_data.ravel(), n, k)
                
            # Save the output data in the data tape
            tape_idx = tape_push((out_data,))
            
            # Return the output data
            return out_data, tape_idx
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # Extract the output data from the saved data
            if saved_data is None:
                # If the output data is not found, raise an error
                raise ValueError("Output data not found in the data tape")

            # Extract the output data from the saved data
            out_data = saved_data[0]
            
            # Compute the gradient of the softmax function
            softmax_gradient(out_data.ravel(), out_grad.ravel(), out_buffer.ravel(), n, k)
            
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
        
        # Extract the number of elements in the input tensor and create an empty array to store the output data
        k = self.data.shape[-1]
        n = self.data.size // k
        
        # Define the forward function
        def forward(*args, **kwargs) -> tuple[np.ndarray, int]:
            # Extract the number of elements in the input tensor
            ndim = self.data.ndim
            
            # Compute the axis for log softmax
            ax = axis % ndim
            
            # If the axis is not the last one, compute log softmax along the specified axis
            if ax != ndim - 1:
                # Compute the maximum value along the specified axis
                m = np.max(self.data, axis=axis, keepdims=True)
                
                # Subtract the maximum value from the input data
                y = self.data - m
                
                # Compute the log sum of exponentials
                logsum = np.log(np.sum(np.exp(y), axis=axis, keepdims=True))
                
                # Compute the log softmax
                out_data = y - logsum
            
            # If the axis is the last one, compute log softmax using the kernel function
            else:
                # Create an empty array to store the output data
                out_data = np.empty_like(self.data)
                
                # Compute the log softmax function
                log_softmax_forward(self.data.ravel(), out_data.ravel(), n, k)
                
            # Save the output data in the data tape
            tape_idx = tape_push((out_data,))
                
            # Return the output data
            return out_data, tape_idx
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # Extract the output data from the saved data
            if saved_data is None:
                # If the output data is not found, raise an error
                raise ValueError("Output data not found in the data tape")
            
            # Extract the output data from the saved data
            out_data = saved_data[0]
            
            # Compute the gradient of the log softmax function
            log_softmax_gradient(out_data.ravel(), out_grad.ravel(), out_buffer.ravel(), n, k)
            
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
            # Initialize the index for the data tape
            tape_idx = -1
            
            # If axis is None, compute the maximum value of the flattened tensor
            if axis is None:
                # Create a buffer to store the maximum value and its index
                buf = np.zeros((1,), dtype=x_data.dtype)
                idx = np.zeros((1,), dtype=np.int64)
                
                # Compute the maximum value of the flattened tensor
                max_flat_forward(x_data.ravel(), buf, idx)
                
                # If keepdims is True, create an output tensor with the same shape as the input tensor
                if keepdims:
                    # Create an output tensor with the same shape as the input tensor
                    out_data = np.full([1]*x_data.ndim, buf[0], dtype=x_data.dtype)
                # If keepdims is False, create an output tensor with the shape of the maximum value
                else:
                    # Create an output tensor with the shape of the maximum value
                    out_data = buf[0]
                    
                # Save the index in the data tape to use it in the backward pass
                tape_idx = tape_push((idx,))
                
            # If axis is not None, compute the maximum value along the specified axis
            else:
                # If axis is not None, compute the maximum value along the specified axis
                out_data = np.max(x_data, axis=axis, keepdims=keepdims)
                
            # Return the computed maximum value and its index
            return out_data, tape_idx
                
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, saved_data: Optional[Tuple[Any, ...]], *args, **kwargs) -> None:
            # If axis is None, compute the gradient for the flattened tensor
            if axis is None:
                # Extract the index from the saved data
                if saved_data is None:
                    # If the index is not found, raise an error
                    raise ValueError("Index not found in the data tape")
                
                # Compute the index of the maximum value
                max_flat_gradient(saved_data[0], np.array([out_grad]).ravel(), out_buffer.ravel())
                
            # If axis is not None, compute the gradient along the specified axis
            else:
                # Initialize the gradient tensor
                expanded = out_grad
                
                # If axis is a tuple, expand the gradient for each axis in the tuple
                if not keepdims:
                    # If keepdims is False, expand the gradient along the specified axis
                    expanded = np.expand_dims(expanded, axis=axis)
                    
                # Create a mask to identify the maximum values
                mask = (self.data == expanded)

                # Count the number of maximum values along the specified axis
                count = np.sum(mask, axis=axis, keepdims=True)
                
                # Avoid division by zero; set count to 1 where it is zero
                grad_x = mask * (expanded / count)
                
                # Accumulate the gradient in the input tensor
                accumulate_gradient(self, grad_x)

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
        
        # Define the forward function
        def forward( x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the square root of the tensor
            return sqrt_forward(x_data), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the square root operation
            sqrt_backward(out_grad, out_buffer, self.data)
        
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
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the mean of the tensor along the specified axis
            return np.mean(x_data, axis=axis, keepdims=keepdims), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # If axis is None, compute the gradient for the flattened tensor
            if axis is None:
                # Create a buffer to store the gradient
                buf = np.zeros((1,), dtype=self.data.dtype)
                
                # Initialize the buffer with the gradient of the output tensor
                buf[0] = out_grad
                inv = 1.0 / self.data.size
                
                # Compute the gradient of the mean operation
                mean_flat_backward(buf, out_buffer.ravel(), inv)
                
            # If axis is not None, compute the gradient along the specified axis
            else:
                # Initialize the gradient tensor
                grad = out_grad
                
                # If keepdims is False, expand the gradient for each axis in the tuple
                if not keepdims:
                    # Expand the gradient along the specified axis
                    grad = np.expand_dims(grad, axis=axis)

                # Compute the number of elements along the specified axis/axes
                if isinstance(axis, int):
                    # If axis is an integer, compute the number of elements along that axis
                    num_elements_along_axis = self.data.shape[axis]
                    
                elif isinstance(axis, tuple):
                    # If axis is a tuple, compute the number of elements along each axis
                    num_elements_along_axis = np.prod([self.data.shape[ax] for ax in axis])
                else:
                    # If axis is not an integer or a tuple, raise a TypeError
                    raise TypeError("axis must be an int or a tuple of ints")

                # Accumulate the gradient in the input tensor
                accumulate_gradient(self, grad / num_elements_along_axis)

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
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the exponential of the tensor
            return np.exp(x_data), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the exponential operation
            exp_gradient(out_grad.ravel(), self.data.ravel(), out_buffer.ravel())
        
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
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the natural logarithm of the tensor
            return np.log(x_data), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the logarithm operation
            log_gradient(self.data.ravel(), out_grad.ravel(), out_buffer.ravel())
        
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
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Compute the transpose of the tensor
            return x_data.transpose(axes), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, *args, **kwargs) -> None:
            # Invert the axes to match the original tensor
            inv_axes = np.argsort(axes)
            
            # Transpose the gradient to match the original tensor
            grad_x = out_grad.transpose(inv_axes)
            
            # Accumulate the gradient in the input tensor
            accumulate_gradient(self, grad_x)
        
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
                
            # Save the mask in the data tape to use it in the backward pass
            tape_idx = tape_push((mask_flat,))
        
            # Return the output data
            return out_data, tape_idx
        
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


    def clip(self, min_val: float, max_val: float) -> 'Tensor':
        """
        Method to clip the tensor within the specified range
        
        Parameters:
        - min_val (float): Minimum value to clip
        - max_val (float): Maximum value to clip
        
        Returns:
        - Tensor: Clipped tensor
        """
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Create an output tensor with the same shape as the input tensor
            out_data = np.empty_like(x_data)
        
            # Clip the values of the tensor to the specified range
            clip_forward(self.data.ravel(), min_val, max_val, out_data.ravel())
            
            # Return the clipped tensor
            return out_data, -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Compute the gradient of the clipping operation
            clip_gradient(self.data.ravel(), out_grad.ravel(), out_buffer.ravel(), min_val, max_val)
                
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
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Squeeze the tensor along the specified axis
            return np.squeeze(x_data, axis=axis), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, *args, **kwargs) -> None:        
            # Unsqueeze the gradient along the same axis to match the original shape
            if axis is None:
                # For None case, we need to restore all squeezed dims
                grad_squeezed = out_grad
                original_shape = x.data.shape
                for dim in sorted([i for i, size in enumerate(original_shape) if size == 1], reverse=True):
                    grad_squeezed = np.expand_dims(grad_squeezed, axis=dim)
            else:
                # For specific axis case
                grad_squeezed = np.expand_dims(out_grad, axis=axis)
            
            # Update the gradient of the input tensor
            accumulate_gradient(self, grad_squeezed)
        
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
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Reshape the tensor to the specified new shape
            return x_data.reshape(shape), -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, *args, **kwargs) -> None:        
            # Reshape the gradient to match the original tensor shape
            grad_back = out_grad.reshape(self.data.shape)
            
            # Accumulate the gradient in the input tensor
            accumulate_gradient(self, grad_back)
        
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
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # If the axis is None, flatten the tensor and repeat
            if axis is None:
                # Flatten the tensor
                out_data = np.empty(self.data.size * repeats, dtype=self.data.dtype)
                
                # Repeat the flattened tensor
                repeat_forward(self.data.ravel(), repeats, out_data)
                
            # If the axis is specified, repeat along that axis
            else:
                # Repeat the tensor along the specified axis
                out_data = np.repeat(self.data, repeats, axis=axis)
                
            # Return the repeated tensor
            return out_data, -1
        
        # Define the backward function
        def backward(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # If the axis is None, compute the gradient for the flattened tensor
            if axis is None:
                # Compute the gradient of the repeated tensor
                repeat_gradient(out_grad.ravel(), repeats, out_buffer.ravel())
                
            # If the axis is specified, compute the gradient along that axis
            else:
                # Reduce the gradient along the specified axis
                grad_unrepeated = np.add.reduce(
                    out_grad.reshape(
                        *(self.data.shape[:axis]), self.data.shape[axis], repeats, *self.data.shape[axis+1:]
                    ), axis = axis+1
                )
                
                # Accumulate the gradient
                accumulate_gradient(self, grad_unrepeated)
        
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
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Extract the padding widths for each dimension and the input tensor shape
            (_, _), (pt, pb), (pl, pr), (_, _) = pad_width
            batch_size, height, width, channels = self.data.shape
            
            # Create the output tensor with the new shape
            out_data = np.empty((batch_size, height + pt + pb, width + pl + pr, channels), dtype=self.data.dtype)
            
            # Perform the padding operation
            pad_forward(self.data, pt, pb, pl, pr, out_data)
            
            # Return the padded tensor
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
        
        # Define the forward function
        def forward(x_data: np.ndarray, kernel_data: np.ndarray) -> tuple[np.ndarray, int]:
            # Perform the convolution operation
            return conv_2d_forward(x_data, kernel_data, stride), -1
        
        # Define the backward function for the input tensor
        def backward_x(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Backprop the gradient through the convolution operation
            conv_2d_backward_x(out_grad=out_grad, out_buffer=out_buffer, kernel_data=kernel.data, stride=stride)
            
        # Define the backward function for the kernel
        def backward_w(out_grad: np.ndarray, out_buffer: np.ndarray, *args, **kwargs) -> None:
            # Backprop the gradient through the convolution operation
            conv_2d_backward_w(out_grad=out_grad, out_buffer=out_buffer, x_data=self.data, stride=stride)
        
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
        
        # Define the stride values and initialize the indices for max pooling
        stride_height, stride_width = stride
        
        # Define the forward function
        def forward(x_data: np.ndarray) -> tuple[np.ndarray, int]:        
            # Extract the input dimensions
            batch_size, height, width, channels = self.data.shape
            kernel_height, kernel_width = kernel_size
            
            # Compute the output dimensions
            out_height = (height - kernel_height) // stride_height + 1
            out_width = (width - kernel_width) // stride_width + 1
            
            # Check if the kernel or stride is too large for the input size
            if out_height < 1 or out_width < 1:
                raise ValueError("Kernel size or stride too large for input size.")

            # Create the output array
            out_data = np.empty((batch_size, out_height, out_width, channels), dtype=self.data.dtype)
            
            # Initialize the indices for max pooling
            arg_i = np.zeros_like(out_data, dtype=np.int32)
            arg_j = np.zeros_like(out_data, dtype=np.int32)

            # Perform the max pooling operation
            max_pool_2d_forward(self.data, kernel_height, kernel_width, stride_height, stride_width, out_data, arg_i, arg_j)
            
            # Save the indices in the data tape to use them in the backward pass
            tape_idx = tape_push((arg_i, arg_j))
            
            # Return the output data
            return out_data, tape_idx
        
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
            # Extract the rank of the first tensor
            rank = tensors_list[0].ndim
            
            # Rank 0
            if rank == 1:
                # Concatenate 1D tensors
                out, offsets = concat_1d_forward(tensors_list)
                
            # Rank 1
            elif rank == 2:
                # Concatenate 2D tensors
                out, offsets = concat_2d_forward(tensors_list, axis)
            
            # Invalid rank  
            else:
                # Raise an error for unsupported tensor dimensions
                raise ValueError(f"Unsupported tensor dimension for concatenation: {rank}")
            
            # Save the offsets in the data tape to use it in the backward pass
            tape_idx = tape_push((offsets,))
                
            # Return the concatenated tensor
            return out, tape_idx
        
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