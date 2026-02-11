from typing import Literal, Optional

from .base import LossFn
from ..core import Tensor, ModuleOutput
from ..core.utils.constants import EPSILON


class BinaryCrossEntropy(LossFn):
            
    ### Magic methods ###
    
    def __init__(
        self, 
        from_logits: bool = False, 
        reduction: Optional[Literal["sum", "mean"]] = "mean",
        from_sequence: bool = False
    ) -> None:
        """
        Initialize the binary cross-entropy loss function.
        
        Parameters:
        - from_logits (bool): Whether the input is logits or probabilities. Default is False
        - reduction (Literal["sum", "mean"]): Reduction method. Default is "mean"
        - from_sequence (bool): If True, the loss function is applied to sequences. Default is False
        """
        
        # Store the attributes
        self.from_logits = from_logits
        self.reduction = reduction
        self.from_sequence = from_sequence
        

    def __call__(self, y_true: Tensor, y_pred: Tensor, **aux: Tensor) -> ModuleOutput:
        """
        Compute the binary cross-entropy loss.
        
        Parameters:
        - y_true (Tensor): True target variable
        - y_pred (Tensor): Predicted target variable
        - **aux (Tensor): Auxiliary tensors (unused, accepted for interface compatibility).
        
        Returns:
        - ModuleOutput: the binary cross-entropy loss as a ModuleOutput containing a single tensor
        """
        
        # If from_sequence is True, reshape the tensors for sequence-to-sequence loss
        if self.from_sequence:
            # Get the features size from the predictions
            features_size = y_pred.shape[-1]
            
            # Reshape predictions from (B, S, F) to (B*S, F)
            y_pred = y_pred.reshape((-1, features_size))
            
            # Reshape targets from (B, S) to (B*S,)
            y_true = y_true.reshape((-1,))
        
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
            return ModuleOutput(loss.sum())
        elif self.reduction == "mean":
            # Return the mean loss
            return ModuleOutput(loss.mean())

        # Return the per-sample loss
        return ModuleOutput(loss)