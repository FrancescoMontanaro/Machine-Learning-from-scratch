import numpy as np
from typing import Literal, Optional

from .base import LossFn
from ..core import Tensor, ModuleOutput
from ..core.utils.constants import EPSILON


class KullbackLeiblerDivergence(LossFn):
    
    ### Magic methods ###
    
    def __init__(
        self,
        reduction: Optional[Literal["mean", "sum"]] = "mean",
        from_sequence: bool = False,
        from_logits: bool = True
    ) -> None:
        """
        Class constructor for Kullback-Leibler divergence loss function.
        
        Parameters:
        - reduction (str): Specifies the reduction to apply to the output. Default is "mean".
        - from_sequence (bool): If True, the loss function is applied to sequences. Default is False.
        - from_logits (bool): If True, y_pred is expected to be logits. Default is True.
        """

        # Store the attributes
        self.reduction = reduction
        self.from_sequence = from_sequence
        self.from_logits = from_logits


    def __call__(self, y_true: Tensor, y_pred: Tensor, **aux: Tensor) -> ModuleOutput:
        """
        Compute the Kullback-Leibler divergence loss.
        
        Parameters:
        - y_true (Tensor): Target distribution (probabilities or class indices).
        - y_pred (Tensor): Predicted logits or probabilities.
        - **aux (Tensor): Auxiliary tensors (unused, accepted for interface compatibility).

        Returns:
        - ModuleOutput: the KL divergence loss as a ModuleOutput containing a single tensor
        """

        # If from_sequence is True, reshape the tensors for sequence-to-sequence loss
        if self.from_sequence:
            # Get the features size from the predictions
            features_size = y_pred.shape[-1]

            # Reshape predictions from (B, S, F) to (B*S, F)
            y_pred = y_pred.reshape((-1, features_size))

            # Reshape targets to match predictions
            if len(y_true.shape) == len(y_pred.shape):
                # Targets are distributions
                y_true = y_true.reshape((-1, features_size))
            else:
                # Targets are class indices
                y_true = y_true.reshape((-1,))

        # Convert target to one-hot if needed
        if len(y_true.shape) == len(y_pred.shape):
            # Target is already a distribution
            target_dist = y_true
        else:
            # Extract the number of classes
            num_classes = y_pred.shape[-1]

            # Convert target to one-hot encoding
            target_dist = Tensor(np.eye(num_classes)[y_true.data.astype(int)], requires_grad=False)

        # Compute log probabilities for predictions
        if self.from_logits:
            log_probs = y_pred.log_softmax(axis=-1)
        else:
            log_probs = y_pred.clip(EPSILON, 1 - EPSILON).log()

        # Compute log probabilities for targets
        target_log_probs = target_dist.clip(EPSILON, 1 - EPSILON).log()

        # Compute the KL divergence
        loss = (target_dist * (target_log_probs - log_probs)).sum(axis=-1)

        # Apply reduction
        if self.reduction == "mean":
            # Return the mean loss
            return ModuleOutput(loss.mean())
        elif self.reduction == "sum":
            # Return the sum loss
            return ModuleOutput(loss.sum())

        # Return the per-sample loss
        return ModuleOutput(loss)
