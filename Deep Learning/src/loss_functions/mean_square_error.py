from .base import LossFn
from ..core import Tensor


class MeanSquareError(LossFn):
    
    ### Magic methods ###

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute the mean squared error loss.
        
        Parameters:
        - y_true (Tensor): True target variable
        - y_pred (Tensor): Predicted target variable
        
        Returns:
        - Tensor: Loss value
        """
        
        # Compute and return the loss
        return ((y_true - y_pred) ** 2).mean()