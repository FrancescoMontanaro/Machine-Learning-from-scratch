import numpy as np
from typing import Literal

from .base import LossFn
from ..activations import Sigmoid



class BinaryCrossEntropy(LossFn):
            
    ### Magic methods ###
    
    def __init__(self, from_logits: bool = False, reduction: Literal["sum", "sum_over_batch_size"] = "sum_over_batch_size") -> None:
        """
        Initialize the binary cross-entropy loss function.
        
        Parameters:
        - from_logits (bool): Whether the input is logits or probabilities. Default is False
        - reduction (Literal["sum", "sum_over_batch_size"]): Reduction method. Default is "sum_over_batch_size"
        """
        
        # Store the attributes
        self.from_logits = from_logits
        self.reduction = reduction
        

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
        
        if self.from_logits:
            # Convert logits to probabilities
            y_pred = Sigmoid()(y_pred)
        else:
            # Clip values for stability
            y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Compute and return the binary cross-entropy loss
        loss = float(-np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / batch_size)
        
        # Apply the reduction method
        if self.reduction == "sum":
            return loss
        else:
            return loss / batch_size


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
        
        if self.from_logits:
            # Compute probabilities from logits
            prob = Sigmoid()(y_pred)
            # Gradient with respect to logits: (sigmoid(z) - y)
            grad = (prob - y_true) / batch_size
            
        else:
            # Return the gradient
            grad = - (y_true / y_pred - (1 - y_true) / (1 - y_pred)) / batch_size
            
        return grad
