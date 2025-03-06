import numpy as np

from .base import Activation
    
    
class Sigmoid(Activation):
    
    ### Magic methods ###
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the output of the sigmoid activation function.
        
        Parameters:
        - x (np.ndarray): Input to the activation function
        
        Returns:
        - np.ndarray: Output of the activation function
        """
        
        # Compute the sigmoid
        return 1 / (1 + np.exp(-x))


    ### Public methods ###

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the sigmoid activation function.
        
        Parameters:
        - x (np.ndarray): Input to the activation function
        
        Returns:
        - np.ndarray: Derivative of the activation function
        """
        
        # Compute the derivative of the sigmoid
        return self(x) * (1 - self(x))