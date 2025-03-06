import numpy as np

from .base import Activation


class Tanh(Activation):
        
    ### Magic methods ###
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the output of the hyperbolic tangent activation function.
        
        Parameters:
        - x (np.ndarray): Input to the activation function
        
        Returns:
        - np.ndarray: Output of the activation function
        """
        
        # Compute the hyperbolic tangent
        return np.tanh(x)


    ### Public methods ###

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the hyperbolic tangent activation function.
        
        Parameters:
        - x (np.ndarray): Input to the activation function
        
        Returns:
        - np.ndarray: Derivative of the activation function
        """
        
        # Compute the derivative of the hyperbolic tangent
        return 1 - np.tanh(x) ** 2