from .base import LossFn
from ..core import Tensor, ModuleOutput


class MeanSquareError(LossFn):
    
    ### Magic methods ###

    def __call__(self, y_true: Tensor, y_pred: Tensor, **aux: Tensor) -> ModuleOutput:
        """
        Compute the mean squared error loss.
        
        Parameters:
        - y_true (Tensor): True target variable
        - y_pred (Tensor): Predicted target variable
        
        Returns:
        - ModuleOutput: the mean squared error loss as a ModuleOutput containing a single tensor
        """
        
        # Compute and return the loss
        return ModuleOutput(((y_true - y_pred) ** 2).mean())
