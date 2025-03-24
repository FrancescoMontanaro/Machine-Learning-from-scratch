from typing import Literal, Optional

from .base import LossFn
from ..core import Tensor
from ..core.utils.constants import *
    
    
class CategoricalCrossEntropy(LossFn):
        
    ### Magic methods ###
    
    def __init__(self, from_logits: bool = False, reduction: Optional[Literal["sum", "mean"]] = "mean") -> None:
        """
        Initialize the cross-entropy loss function.
        
        Parameters:
        - from_logits (bool): Whether the input is logits or probabilities. Default is False
        - reduction (Literal["sum", "mean"]): Reduction method. Default is "mean"
        """
        
        # Store the attributes
        self.from_logits = from_logits
        self.reduction = reduction
        

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute the cross-entropy loss.
        
        Parameters:
        - y_true (Tensor): True target variable
        - y_pred (Tensor): Predicted target variable
        
        Returns:
        - Tensor: Loss value
        """
        
        # Check if the input is logits
        if self.from_logits:
            # Convert logits to probabilities
            y_pred_log = y_pred.log_softmax(axis=1)
        else:
            # Clip values for numerical stability
            y_pred_log = y_pred.clip(EPSILON, 1 - EPSILON).log()

        # Compute the loss
        loss = - (y_true * y_pred_log).sum(axis=-1)
        
        # Apply the reduction method
        if self.reduction == "sum":
            # Return the sum loss
            return loss.sum()
        elif self.reduction == "mean":
            # Return the mean loss
            return loss.mean()
        else:
            # Return the per-sample loss
            return loss