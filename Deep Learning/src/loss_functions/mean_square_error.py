import numpy as np

from .base import LossFn


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