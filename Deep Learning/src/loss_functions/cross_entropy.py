import numpy as np
from typing import Literal, Optional

from .base import LossFn
from ..core import Tensor
from ..core.utils.constants import EPSILON


class CrossEntropy(LossFn):
    
    def __init__(
        self, 
        reduction: Optional[Literal["mean", "sum"]] = "mean", 
        label_smoothing: float = 0.0,
        from_sequence: bool = False,
        from_logits: bool = True
    ) -> None:
        """
        Class constructor for CrossEntropy loss function.
        
        Parameters:
        - reduction (str): Specifies the reduction to apply to the output. Default is "mean".
        - label_smoothing (float): If greater than 0, applies label smoothing. Default is 0.0.
        - from_sequence (bool): If True, the loss function is applied to sequences. Default is False.
        - from_logits (bool): If True, y_pred is expected to be logits. Default is True.
        """
        
        # Store the attributes
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.from_sequence = from_sequence
        self.from_logits = from_logits


    def __call__(self, y_true: Tensor, y_pred: Tensor, **aux: Tensor) -> Tensor:
        """
        Compute the cross-entropy loss.
        
        Parameters:
        - y_true (Tensor): Target labels.
        - y_pred (Tensor): Predicted logits or probabilities.
        - **aux (Tensor): Auxiliary tensors (unused, accepted for interface compatibility).
        """
        
        # If from_sequence is True, reshape the tensors for sequence-to-sequence loss
        if self.from_sequence:
            # Get the features size from the predictions
            features_size = y_pred.shape[-1]
            
            # Reshape predictions from (B, S, F) to (B*S, F)
            y_pred = y_pred.reshape((-1, features_size))
            
            # Reshape targets from (B, S) to (B*S,)
            y_true = y_true.reshape((-1,))
        
        # Convert target to one-hot if needed
        if len(y_true.shape) == len(y_pred.shape):
            # Target is already one-hot
            target_one_hot = y_true
        else:
            # Extract the number of classes
            num_classes = y_pred.shape[-1]
            
            # Convert target to one-hot encoding
            target_one_hot = Tensor(np.eye(num_classes)[y_true.data.astype(int)], requires_grad=False)

        # Apply label smoothing
        if self.label_smoothing > 0:
            # Smooth the labels
            smoothing_value = self.label_smoothing / (y_pred.shape[-1] - 1)
            
            # Apply label smoothing
            target_one_hot = target_one_hot * (1 - self.label_smoothing - smoothing_value) + smoothing_value

        # Compute log probabilities
        if self.from_logits:
            log_probs = y_pred.log_softmax(axis=-1)
        else:
            log_probs = y_pred.clip(EPSILON, 1 - EPSILON).log()
        
        # Compute negative log likelihood
        loss = - (target_one_hot * log_probs).sum(axis=-1)
        
        # Apply reduction
        if self.reduction == "mean":
            # Return the mean loss
            return loss.mean()
        elif self.reduction == "sum":
            # Return the sum loss
            return loss.sum()
        
        # Return the per-sample loss
        return loss
