import numpy as np

from .base import Activation
    

class Softmax(Activation):
        
    ### Magic methods ###
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the output of the softmax activation function.
        
        Parameters:
        - x (np.ndarray): Input to the activation function
        
        Returns:
        - np.ndarray: Output of the activation function
        """
        
        # Compute the softmax
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        
        # Normalize the output
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


    ### Public methods ###

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the softmax activation function.
        
        Parameters:
        - x (np.ndarray): Input to the activation function
        
        Returns:
        - np.ndarray: Derivative of the activation function
        """
        
        # Compute the derivative of the softmax
        softmax_x = self(x)
        
        # Compute the Jacobian matrix
        return softmax_x * (1 - softmax_x)