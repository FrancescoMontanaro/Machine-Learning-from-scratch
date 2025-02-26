import numpy as np
from functools import wraps
from typing import Optional, Callable

from ..optimizers import Optimizer


class Layer:
    
    ### Magic methods ###
    
    def __init__(self, name: Optional[str] = None) -> None:
        """
        Class constructor of the abstract class Layer
        """
        
        self.name = name # Name of the layer
        self.training = True # Flag to check if the layer is in training mode
        self.initialized = False # Flag to check if the layer is initialized
        self.optimizer: Optional[Optimizer] = None # Instance of the optimizer to be used for updating the parameters
    
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Abstract method to set the input shape of the layer.
        
        Parameters:
        - x (np.ndarray): Input data
        
        Returns:
        - np.ndarray: Output of the layer after the forward pass
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method '__call__' is not implemented.")
    
    
    ### Public methods ###
    
    @staticmethod
    def check_initialization(func: Callable) -> Callable:
        """
        Decorator to check if the Layer parameters are initialized before calling the function.
        
        Parameters:
        - func (Callable): Function to be wrapped
        
        Returns:
        - Callable: Wrapper function
        
        Raises:
        - ValueError: If the layer parameters are not initialized
        """
        
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            """
            Wrapper function to check if the layer is initialized before calling the function.
            
            Parameters:
            - self: Instance of the class
            - *args: Variable length argument list
            - **kwargs: Arbitrary keyword arguments
            """
            
            # Check if the layer is initialized
            if not self.initialized:
                # Raise an error if the layer is not initialized
                raise ValueError("Layer is not initialized. Please call the layer with input data.")
            
            # Call the function
            return func(self, *args, **kwargs)
        
        # Return the wrapper function
        return wrapper
    
    
    def parameters(self) -> dict:
        """
        Abstract method to return the parameters of the layer. 
        The implementation should be done in the child classes
        
        Returns:
        - dict: Parameters of the layer.
        """
        
        # If the layer does not have any parameters, return an empty dictionary
        return {}

    
    @check_initialization
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Abstract method to compute the output of the layer. 
        The implementation should be done in the child classes
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        
        Returns:
        - np.ndarray: Output of the layer
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method 'forward' is not implemented.")
    
    
    @check_initialization
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Abstract method to compute the gradient of the loss with respect to the input of the layer. 
        The implementation should be done in the child classes
        
        Parameters:
        - grad_output (np.ndarray): Gradient of the loss with respect to the output of the layer
        
        Returns:
        - np.ndarray: Gradient of the loss with respect to the input of the layer
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        raise NotImplementedError("The method 'backward' is not implemented.")

    
    @check_initialization
    def output_shape(self) -> tuple:
        """
        Abstract method to return the output shape of the layer. 
        The implementation should be done in the child classes
        
        Returns:
        - tuple: The shape of the output of the layer
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        raise NotImplementedError("The method 'output_shape' is not implemented.")
    
    
    @check_initialization
    def count_params(self) -> int:
        """
        Method to count the number of parameters in the layer
        
        Returns:
        - int: Number of parameters in the layer
        """
        
        return 0
    
    
    def init_params(self, *args, **kwargs) -> None:
        """
        Abstract method to initialize the parameters of the layer. 
        The implementation should be done in the child classes
        """
        
        # Set the flag to True to indicate that the layer is initialized
        self.initialized = True