from typing import Literal, Optional

from .base import LossFn
from ..core import Tensor
from ..core.utils.constants import *


class BinaryCrossEntropy(LossFn):
            
    ### Magic methods ###
    
    def __init__(self, from_logits: bool = False, reduction: Optional[Literal["sum", "mean"]] = "mean") -> None:
        """
        Initialize the binary cross-entropy loss function.
        
        Parameters:
        - from_logits (bool): Whether the input is logits or probabilities. Default is False
        - reduction (Literal["sum", "mean"]): Reduction method. Default is "mean"
        """
        
        # Store the attributes
        self.from_logits = from_logits
        self.reduction = reduction
        

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute the binary cross-entropy loss.
        
        Parameters:
        - y_true (Tensor): True target variable
        - y_pred (Tensor): Predicted target variable
        
        Returns:
        - Tensor: the binary cross-entropy loss
        """
        
        # Check if the input is logits
        if self.from_logits:
            # Convert logits to probabilities
            y_pred = y_pred.softmax()
        else:
            # Clip values for stability
            y_pred = y_pred.clip(EPSILON, 1 - EPSILON)
        
        # Compute the binary cross-entropy loss
        loss = - (y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log())
        
        # Apply the reduction method
        if self.reduction == "sum":
            # Return the sum loss
            return loss.sum()
        elif self.reduction == "mean":
            # Return the mean loss
            return loss.mean()

        # Return the per-sample loss
        return loss