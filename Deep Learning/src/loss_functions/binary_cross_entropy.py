import numpy as np

from .base import LossFn


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
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
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
