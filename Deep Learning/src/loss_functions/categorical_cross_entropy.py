from typing import Literal, Optional

from .base import LossFn
from ..core import Tensor
from ..core.utils.constants import *
    
class CategoricalCrossEntropy(LossFn):
        
    ### Magic methods ###
    
    def __init__(
        self, 
        from_logits: bool = False, 
        reduction: Optional[Literal["sum", "mean"]] = "mean",
        from_sequence: bool = False
    ) -> None:
        """
        Initialize the cross-entropy loss function.
        
        Parameters:
        - from_logits (bool): Whether the input is logits or probabilities. Default is False
        - reduction (Literal["sum", "mean"]): Reduction method. Default is "mean"
        - from_sequence (bool): If True, the loss function is applied to sequences. Default is False
        """
        
        # Store the attributes
        self.from_logits = from_logits
        self.reduction = reduction
        self.from_sequence = from_sequence
        

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute the cross-entropy loss.
        
        Parameters:
        - y_true (Tensor): True target variable
        - y_pred (Tensor): Predicted target variable
        
        Returns:
        - Tensor: Loss value
        """
        
        # If from_sequence is True, reshape the tensors for sequence-to-sequence loss
        if self.from_sequence:
            # Get the features size from the predictions
            features_size = y_pred.shape()[-1]
            
            # Reshape predictions from (B, S, F) to (B*S, F)
            y_pred = y_pred.reshape((-1, features_size))
            
            # Reshape targets from (B, S) to (B*S,)
            y_true = y_true.reshape((-1,))
        
        # Check if the input is logits
        if self.from_logits:
            # Convert logits to log probabilities
            y_pred_log = y_pred.log_softmax(axis=-1)
        else:
            # Ensure the probabilities sum to 1 
            y_pred = y_pred / y_pred.sum(axis=-1, keepdims=True)
            
            # Clip values for numerical stability and compute log probabilities
            y_pred_log = y_pred.clip(EPSILON, 1 - EPSILON).log()

        # Compute the loss
        loss = - (y_true * y_pred_log).sum(axis=-1)
        
        # Apply the reduction method
        if self.reduction == "sum":
            # Return the sum loss
            loss = loss.sum()
        elif self.reduction == "mean":
            # Return the mean loss
            loss = loss.mean()
        
        # Return the loss
        return loss