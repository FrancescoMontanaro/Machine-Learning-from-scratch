import numpy as np
from typing import Optional, Union, Tuple, Callable

from .functional.operators import *
from .functional.functions import *
from .functional.activations import *
from .utils.context_manager import _NO_GRAD


class Tensor:
    
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

        # Store the data, gradient, and the flag to compute the gradient
        self.data = np.array(data).astype(dtype)
            
        # Import the global flag to disable gradient computation
        global _NO_GRAD
        
        # Initialize the control variables
        self.is_parameter = is_parameter # Flag to indicate if the tensor is a trainable parameter
        self.requires_grad = False if _NO_GRAD else requires_grad # Flag to indicate if the gradient needs to be computed for the tensor
        self.grad = None # Gradient of the tensor with respect to the loss
        self._backward: Callable = lambda: None # Function to backpropagate the gradient
        self._prev: set['Tensor'] = set() # Set to store the previous tensors in the computation graph


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
        
        Raises:
        - AssertionError: If the object to be added is not a tensor
        """
        
        # Assert that the object to be added is a tensor
        assert isinstance(other, (int, float, np.ndarray, Tensor)), "The object to be added should be a tensor, integer, float, or numpy array."
        
        # Check if the object is an integer or a float
        if not isinstance(other, Tensor):
            # Convert the integer or float to a tensor
            other = Tensor(
                data = np.array(other, dtype=self.data.dtype) if isinstance(other, (int, float)) else other, 
                requires_grad = False
            )

        # Compute and return the sum of the two tensors
        return add(self, other)
    
    
    def __sub__(self, other: Union[int, float, np.ndarray, 'Tensor']) -> 'Tensor':
        """
        Method to subtract two tensors
        
        Parameters:
        - other (Union[int, float, np.ndarray, Tensor]): Object to be subtracted from the current tensor
        
        Returns:
        - Tensor: Tensor containing the difference of the two tensors
        
        Raises:
        - AssertionError: If the object to be subtracted is not a tensor
        """
        
        # Assert that the object to be subtracted is a tensor
        assert isinstance(other, (int, float, np.ndarray, Tensor)), "The object to be subtracted should be a tensor, integer, float, or numpy array."
        
        # Check if the object is an integer or a float
        if not isinstance(other, Tensor):
            # Convert the integer or float to a tensor
            other = Tensor(
                data = np.array(other, dtype=self.data.dtype) if isinstance(other, (int, float)) else other, 
                requires_grad = False
            )
        
        # Compute and return the difference of the two tensors
        return sub(self, other)
    

    def __mul__(self, other: Union[int, float, np.ndarray, 'Tensor']) -> 'Tensor':
        """
        Method to multiply two tensors
        
        Parameters:
        - other (Union[int, float, Tensor]): Object to be multiplied with the current tensor
        
        Returns:
        - Tensor: Tensor containing the product of the two tensors
        
        Raises:
        - AssertionError: If the object to be multiplied is not a tensor
        """
        
        # Assert that the object to be multiplied is a tensor
        assert isinstance(other, (int, float, np.ndarray, Tensor)), "The object to be multiplied should be a tensor, integer, float, or numpy array."
        
        # Check if the object is an integer or a float
        if not isinstance(other, Tensor):
            # Convert the integer or float to a tensor
            other = Tensor(
                data = np.array(other, dtype=self.data.dtype) if isinstance(other, (int, float)) else other, 
                requires_grad = False
            )
        
        # Compute and return the product of the two tensors
        return mul(self, other)
    
    
    def __truediv__(self, other: Union[int, float, np.ndarray, 'Tensor']) -> 'Tensor':
        """
        Method to divide two tensors
        
        Parameters:
        - other (Union[int, float, np.ndarray, Tensor]): Object to be divided by the current tensor
        
        Returns:
        - Tensor: Tensor containing the division of the two tensors
        
        Raises:
        - AssertionError: If the object to be divided is not a tensor
        """
        
        # Assert that the object to be divided is a tensor
        assert isinstance(other, (int, float, np.ndarray, Tensor)), "The object to be divided should be a tensor, integer, float, or numpy array."
        
        # Check if the object is an integer or a float
        if not isinstance(other, Tensor):
            # Convert the integer or float to a tensor
            other = Tensor(
                data = np.array(other, dtype=self.data.dtype) if isinstance(other, (int, float)) else other, 
                requires_grad = False
            )
        
        # Compute the division of the two tensors
        return div(self, other)


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
        
        # Assert that the object to be multiplied is a tensor
        assert isinstance(other, Tensor), "The object to be multiplied should be a tensor."
        
        # Compute the matrix multiplication of the two tensors
        return mat_mul(self, other)
    
    
    def __radd__(self, other: Union[int, float, np.ndarray]) -> 'Tensor':
        """
        Method to implement the reverse addition so that expressions like (1 + tensor) can be evaluated.
        
        Parameters:
        - other (Union[int, float, np.ndarray]): A constant value on the left-hand side.
        
        Returns:
        - Tensor: A new Tensor representing (other + self.data).
        
        Raises:
        - AssertionError: If the object to be added is not an integer or a float.
        """
        
        # Assert that the object to be added is an integer or a float
        assert isinstance(other, (int, float, np.ndarray)), "The object to be added should be an integer or a float."
        
        # Convert the constant to a Tensor (with requires_grad=False)
        other_tensor = Tensor(
            data = np.array(other, dtype=self.data.dtype) if isinstance(other, (int, float)) else other, 
            requires_grad = False
        )
            
        # Compute the sum of the two tensors
        return self.__add__(other_tensor)
    
    
    def __rsub__(self, other: Union[int, float, np.ndarray]) -> 'Tensor':
        """
        Method to implement the reverse subtraction so that expressions like (1 - tensor) can be evaluated.
        
        Parameters:
        - other (int, float, np.ndarray): A constant value on the left-hand side.
        
        Returns:
        - Tensor: A new Tensor representing (other - self.data).
        
        Raises:
        - AssertionError: If the object to be subtracted is not an integer or a float.
        """
        
        # Assert that the object to be subtracted is an integer or a float
        assert isinstance(other, (int, float, np.ndarray)), "The object to be subtracted should be an integer or a float."
        
        # Convert the constant to a Tensor (with requires_grad=False)
        other_tensor = Tensor(
            data = np.array(other, dtype=self.data.dtype) if isinstance(other, (int, float)) else other, 
            requires_grad = False
        )
            
        # Compute the difference of the two tensors
        return -self.__sub__(other_tensor)
    
    
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
        
        # Assert that the power is an integer or a float
        assert isinstance(power, (int, float)), "The power should be an integer or a float."
        
        # Compute and return the power of the tensor
        return pow(self, power)
    
    
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
        
        # Compute and return the sliced tensor
        return get_item(self, key)
    

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
            self.grad = np.ones_like(self.data)
            
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
    
    
    ###########################
    ####### Activations #######
    ###########################
    
    def sigmoid(self) -> 'Tensor':
        """
        Method to compute the sigmoid of the tensor
        
        Returns:
        - Tensor: Tensor containing the sigmoid of the current tensor
        """
        
        # Compute and return the sigmoid of the tensor
        return sigmoid(self)
    
    
    def relu(self) -> 'Tensor':
        """
        Method to compute the ReLU of the tensor
        
        Returns:
        - Tensor: Tensor containing the ReLU of the current tensor
        """
        
        # Compute and return the ReLU of the tensor
        return relu(self)
    
    
    def tanh(self) -> 'Tensor':
        """
        Method to compute the hyperbolic tangent of the tensor
        
        Returns:
        - Tensor: Tensor containing the hyperbolic tangent of the current tensor
        """
        
        # Compute and return the hyperbolic tangent of the tensor
        return tanh(self)
    
    
    def softmax(self, axis: int = -1) -> 'Tensor':
        """
        Method to compute the softmax of the tensor
        
        Parameters:
        - axis (int): Axis along which the softmax needs to be computed
        
        Returns:
        - Tensor: Tensor containing the softmax of the current tensor
        
        Raises:
        - AssertionError: If the axis is not an integer
        """
        
        # Assert that the axis is an integer
        assert isinstance(axis, int), "The axis should be an integer."
        
        # Compute and return the softmax of the tensor
        return softmax(self, axis=axis)
    
    
    def log_softmax(self, axis: int = -1) -> 'Tensor':
        """
        Method to compute the log softmax of the tensor
        
        Parameters:
        - axis (int): Axis along which the log softmax needs to be computed
        
        Returns:
        - Tensor: Tensor containing the log softmax of the current tensor
        
        Raises:
        - AssertionError: If the axis is not an integer
        """
        
        # Assert that the axis is an integer
        assert isinstance(axis, int), "The axis should be an integer."
        
        # Compute and return the log softmax of the tensor
        return log_softmax(self, axis=axis)
    
    
    ###########################
    ######## Functions ########
    ###########################
    
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """
        Method to compute the sum of the tensor along the specified axis
        
        Parameters:
        - axis (int): Axis along which the sum needs to be computed
        - keepdims (bool): Flag to indicate if the dimensions need to be kept
        
        Returns:
        - Tensor: Tensor containing the sum of the current tensor along the specified axis
        """
        
        # Compute and return the sum of the tensor along the specified axis
        return sum(self, axis=axis, keepdims=keepdims)
    
    
    def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        Method to compute the maximum value of the tensor along the specified axis
        
        Parameters:
        - axis (Optional[Union[int, Tuple[int, ...]]]): Axis along which to compute the maximum value
        - keepdims (bool): Whether to keep the dimensions of the input tensor
        
        Returns:
        - Tensor: Maximum value of the tensor along the specified axis
        """
        
        # Compute and return the maximum value of the tensor along the specified axis
        return max(self, axis=axis, keepdims=keepdims)
   
   
    def sqrt(self) -> 'Tensor':
        """
        Method to compute the square root of the tensor
        
        Returns:
        - Tensor: Tensor containing the square root of the current tensor
        """
        
        # Compute and return the square root of the tensor
        return sqrt(self)
   
   
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        Method to compute the mean of the tensor along the specified axis
        
        Parameters:
        - axis (int): Axis along which to compute the mean
        - keepdims (bool): Whether to keep the dimensions of the input tensor
        
        Returns:
        - Tensor: Mean of the tensor along the specified axis
        """
        
        # Compute and return the mean of the tensor along the specified axis
        return mean(self, axis=axis, keepdims=keepdims)
    
    
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
        
        # Compute and return the variance of the tensor along the specified axis
        return var(self, axis=axis, keepdims=keepdims, ddof=ddof)
   
    
    def exp(self) -> 'Tensor':
        """
        Method to compute the exponential of the tensor
        
        Returns:
        - Tensor: Tensor containing the exponential of the current tensor
        """
        
        # Compute and return the exponential of the tensor
        return exp(self)


    def log(self) -> 'Tensor':
        """
        Method to compute the natural logarithm of the tensor
        
        Returns:
        - Tensor: Tensor containing the natural logarithm of the current tensor
        """
        
        # Compute and return the natural logarithm of the tensor
        return log(self)

     
    def transpose(self, axes: tuple) -> 'Tensor':
        """
        Method to compute the transpose of the tensor
        
        Parameters:
        - axes (tuple): Permutation of the dimensions
        
        Returns:
        - Tensor: Transposed tensor
        """
        
        # Compute and return the transpose of the tensor
        return transpose(self, axes=axes)


    def masked_fill(self, mask: Union[np.ndarray, 'Tensor'], value: float) -> 'Tensor':
        """
        Method to fill the tensor with a value where the mask is True
        
        Parameters:
        - mask (Union[np.ndarray, Tensor]): Mask to fill the tensor
        - value (float): Value to fill the tensor
        
        Returns:
        - Tensor: Tensor with the masked fill operation
        """
        
        # Compute and return the masked fill operation
        return masked_fill(self, mask, value)


    def clip(self, min_val: float, max_val: float) -> 'Tensor':
        """
        Method to clip the tensor within the specified range
        
        Parameters:
        - min_val (float): Minimum value to clip
        - max_val (float): Maximum value to clip
        
        Returns:
        - Tensor: Clipped tensor
        """
        
        # Compute and return the clipped tensor
        return clip(self, min_val, max_val)
    
    
    def gather(self, axis: int, indices: 'Tensor') -> 'Tensor':
        """
        Method to gather the elements from the tensor along the specified axis
        
        Parameters:
        - axis (int): Axis along which to gather the elements
        - indices (Tensor): Indices to gather
        
        Returns:
        - Tensor: Tensor containing the gathered elements
        """
        
        # Compute and return the gathered elements
        return gather(self, indices, axis)
    
    
    def squeeze(self, axis: int) -> 'Tensor':
        """
        Method to squeeze the tensor along the specified axis
        
        Parameters:
        - axis (int): Axis along which to squeeze the tensor
        
        Returns:
        - Tensor: Sequeezed tensor
        """
        
        # Compute and return the sequeezed tensor
        return squeeze(self, axis)
    
    
    def unsqueeze(self, axis: int) -> 'Tensor':
        """
        Method to unsqueeze the tensor along the specified axis
        
        Parameters:
        - axis (int): Axis along which to unsqueeze the tensor
        
        Returns:
        - Tensor: Unsqueezed tensor
        """
        
        # Compute and return the unsqueezed tensor
        return unsqueeze(self, axis)
    
    
    def reshape(self, shape: tuple) -> 'Tensor':
        """
        Method to reshape the tensor
        
        Parameters:
        - shape (tuple): New shape of the tensor
        
        Returns:
        - Tensor: Reshaped tensor
        """
        
        # Compute and return the reshaped tensor
        return reshape(self, shape)
    
    
    def repeat(self, repeats: int, axis: Optional[int] = None) -> 'Tensor':
        """
        Method to repeat the tensor along the specified dimensions
        
        Parameters:
        - repeats (Union[int, Tuple[int, ...]]): Number of times to repeat the tensor
        - axis (Union[int, Tuple[int, ...]]): Axis along which to repeat the tensor
        
        Returns:
        - Tensor: Repeated tensor
        """
        
        # Compute and return the repeated tensor
        return repeat(self, repeats, axis)
    
    
    def pad(self, pad_width: tuple) -> 'Tensor':
        """
        Method to pad the tensor along the specified dimensions
        
        Parameters:
        - pad_width (tuple): Width of the padding
        
        Returns:
        - Tensor: Padded tensor
        """
        
        # Compute and return the padded tensor
        return pad(self, pad_width)
    
    
    def conv_2d(self, kernel: 'Tensor', stride: Tuple[int,int] = (1,1)) -> 'Tensor':
        """
        Method to compute the 2D convolution of the tensor
        
        Parameters:
        - kernel (Tensor): Kernel for the convolution
        - stride (Tuple[int,int]): Stride for the convolution. Default is (1,1)
        
        Returns:
        - Tensor: Tensor containing the 2D convolution of the tensor
        """
        
        # Compute and return the 2D convolution of the tensor
        return conv_2d(self, kernel, stride)
    
    
    def max_pool_2d(self, kernel_size: Tuple[int,int] = (2,2), stride: Tuple[int,int] = (2,2)) -> 'Tensor':
        """
        Method to compute the 2D max pooling of the tensor
        
        Parameters:
        - kernel_size (Tuple[int,int]): Kernel size for the pooling. Default is (2,2)
        - stride (Tuple[int,int]): Stride for the pooling. Default is (2,2)
        
        Returns:
        - Tensor: Tensor containing the 2D max pooling of the tensor
        """
        
        # Compute and return the 2D max pooling of the tensor
        return max_pool_2d(self, kernel_size, stride)