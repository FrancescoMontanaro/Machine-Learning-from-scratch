import numpy as np

from .base import LossFn

epsilon = 1e-12  # Small constant for numerical stability


class MeanSquareError(LossFn):
    
    ### Magic methods ###

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error loss.
        
        Parameters:
        - y_true (np.ndarray): True target variable
        - y_pred (np.ndarray): Predicted target variable
        
        Returns:
        - float: Loss value
        """
        
        # Compute and return the loss
        return float(np.mean((y_true - y_pred) ** 2))


    ### Public methods ###

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to y_pred.
        
        Parameters:
        - y_true (np.ndarray): True target variable
        - y_pred (np.ndarray): Predicted target variable
        
        Returns:
        - np.ndarray: Gradient of the loss with respect to y_pred
        """
        
        # Extract the batch size
        batch_size = y_true.shape[0]
        
        # Return the gradient
        return (2 / batch_size) * (y_pred - y_true)
    
    
class CrossEntropy(LossFn):
        
    ### Magic methods ###

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross-entropy loss.
        
        Parameters:
        - y_true (np.ndarray): True target variable
        - y_pred (np.ndarray): Predicted target variable
        
        Returns:
        - float: Loss value
        """
        
        # Extract the batch size
        batch_size = y_true.shape[0]
        
        # Clip values for numerical stability
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Compute the cross-entropy loss for one-hot encoded labels
        return -np.sum(y_true * np.log(y_pred)) / batch_size


    ### Public methods ###

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to y_pred.
        
        Parameters:
        - y_true (np.ndarray): True target variable
        - y_pred (np.ndarray): Predicted target variable
        
        Returns:
        - np.ndarray: Gradient of the loss with respect to y_pred
        """
        
        # Extract the batch size
        batch_size = y_true.shape[0]

        # Compute the gradient
        return (y_pred - y_true) / batch_size


class BinaryCrossEntropy(LossFn):
            
    ### Magic methods ###

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the binary cross-entropy loss.
        
        Parameters:
        - y_true (np.ndarray): True target variable
        - y_pred (np.ndarray): Predicted target variable
        
        Returns:
        - float: Loss value
        """
        
        # Extract the batch size
        batch_size = y_true.shape[0]
        
        # Clip values for stability
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Compute and return the binary cross-entropy loss
        return float(-np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / batch_size)


    ### Public methods ###

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to y_pred.
        
        Parameters:
        - y_true (np.ndarray): True target variable
        - y_pred (np.ndarray): Predicted target variable
        
        Returns:
        - np.ndarray: Gradient of the loss with respect to y_pred
        """
        
        # Extract the batch size
        batch_size = y_true.shape[0]
        
        # Return the gradient
        return - (y_true / y_pred - (1 - y_true) / (1 - y_pred)) / batch_size
