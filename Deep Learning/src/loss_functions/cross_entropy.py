import numpy as np

from .base import LossFn
    
    
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
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

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