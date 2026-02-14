from typing import Literal, Optional

from .base import LossFn
from ..core import Tensor, ModuleOutput
from ..core.utils.constants import EPSILON


class KullbackLeiblerDivergence(LossFn):
    
    ### Magic methods ###
    
    def __init__(
        self,
        reduction: Optional[Literal["mean", "sum"]] = "mean",
        from_logits: bool = True
    ) -> None:
        """
        Class constructor for Kullback-Leibler divergence loss function.
        
        Parameters:
        - reduction (str): Specifies the reduction to apply to the output. Default is "mean".
        - from_logits (bool): If True, y_pred is expected to be logits. Default is True.
        """

        # Store the attributes
        self.reduction = reduction
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

        # Normalize shapes and convert targets to class distributions.
        y_true, y_pred = self._prepare_classification_tensors(y_true=y_true, y_pred=y_pred)
        target_dist = self._to_class_distribution(y_true=y_true, y_pred=y_pred)

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
