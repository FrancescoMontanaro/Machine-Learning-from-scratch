import numpy as np


class _AbstractActivationFn:
    
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


class ReLU(_AbstractActivationFn):
    
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
    
    
class Sigmoid(_AbstractActivationFn):
    
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
    
    
class Softmax(_AbstractActivationFn):
        
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


class Tanh(_AbstractActivationFn):
        
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