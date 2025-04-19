import numpy as np
from typing import Literal, Optional

from .base import LossFn
from ..core import Tensor
from ..core.utils.constants import *
from ..core.utils.context_manager import _NO_GRAD
from ..core.functional.utils import accumulate_gradient
    
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
            
        if _NO_GRAD: return loss # If gradient computation is disabled, return the loss tensor without a backward function
        
        # Override backward pass when from_logits=True
        # This is to avoid numerical instability when computing the gradient
        if self.from_logits and y_pred.requires_grad:
            # Define the backward function
            def _backward() -> None:
                # Compute softmax
                softmax = np.exp(y_pred_log.data)
                
                # Computhe the gradient
                grad = softmax - y_true.data
                
                # Apply reduction if needed
                if self.reduction == "sum":
                    # Sum reduction
                    grad = grad
                elif self.reduction == "mean":
                    # Mean reduction
                    grad = grad / y_true.shape()[0]
                
                # Accumulate gradient
                accumulate_gradient(y_pred, grad)
            
            # Store the backward function
            loss._backward = _backward
            
            # Store the previous tensors in the computation graph
            loss._prev = {y_pred} if y_pred.requires_grad else set()
        
        # Return the loss
        return loss