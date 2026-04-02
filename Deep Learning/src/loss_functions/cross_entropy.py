from typing import Literal, Optional

from .base import LossFn
from ..core import Tensor, ModuleOutput
from ..core.utils.constants import EPSILON


class CrossEntropy(LossFn):
    
    def __init__(
        self, 
        reduction: Optional[Literal["mean", "sum"]] = "mean", 
        label_smoothing: float = 0.0,
        from_logits: bool = True
    ) -> None:
        """
        Class constructor for CrossEntropy loss function.
        
        Parameters:
        - reduction (str): Specifies the reduction to apply to the output. Default is "mean".
        - label_smoothing (float): If greater than 0, applies label smoothing. Default is 0.0.
        - from_logits (bool): If True, y_pred is expected to be logits. Default is True.
        """
        
        # Store the attributes
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits


    def __call__(self, y_true: Tensor, y_pred: Tensor, **aux: Tensor) -> ModuleOutput:
        """
        Compute the cross-entropy loss.
        
        Parameters:
        - y_true (Tensor): Target labels.
        - y_pred (Tensor): Predicted logits or probabilities.
        - **aux (Tensor): Auxiliary tensors (unused, accepted for interface compatibility).

        Returns:
        - ModuleOutput: the cross-entropy loss as a ModuleOutput containing a single tensor
        """
        
        # Normalize shapes and convert targets to class distributions.
        y_true, y_pred = self._prepare_classification_tensors(y_true=y_true, y_pred=y_pred)
        target_one_hot = self._to_class_distribution(y_true=y_true, y_pred=y_pred)

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
            return ModuleOutput(loss.mean())
        elif self.reduction == "sum":
            # Return the sum loss
            return ModuleOutput(loss.sum())
        
        # Return the per-sample loss
        return ModuleOutput(loss)
