import numpy as np


class Activation:
    
    ### Magic methods ###

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the output of the activation function.

        Parameters:
        - x (np.ndarray): Input to the activation function

        Returns:
        - np.ndarray: Output of the activation function
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method '__call__' is not implemented.")


    ### Public methods ###

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the activation function.

        Parameters:
        - x (np.ndarray): Input to the activation function

        Returns:
        - np.ndarray: Derivative of the activation function
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method 'derivative' is not implemented.")

