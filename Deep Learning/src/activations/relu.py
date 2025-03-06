import numpy as np

from .base import Activation


class ReLU(Activation):
    
    ### Magic methods ###
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the output of the ReLU activation function.
        
        Parameters:
        - x (np.ndarray): Input to the activation function
        
        Returns:
        - np.ndarray: Output of the activation function
        """
        
        # Compute the ReLU
        return np.maximum(0, x)


    ### Public methods ###

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the ReLU activation function.
        
        Parameters:
        - x (np.ndarray): Input to the activation function
        
        Returns:
        - np.ndarray: Derivative of the activation function
        """
        
        # Compute the derivative of the ReLU
        return np.where(x > 0, 1, 0)