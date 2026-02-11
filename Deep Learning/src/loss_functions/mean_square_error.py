from .base import LossFn
from ..core import Tensor, ModuleOutput


class MeanSquareError(LossFn):
    
    ### Magic methods ###
    
    def __init__(self, from_sequence: bool = False) -> None:
        """
        Class constructor for MeanSquareError loss function.
        
        Parameters:
        - from_sequence (bool): If True, the loss function is applied to sequences. Default is False.
        """
        
        # Store the attributes
        self.from_sequence = from_sequence
        

    def __call__(self, y_true: Tensor, y_pred: Tensor, **aux: Tensor) -> ModuleOutput:
        """
        Compute the mean squared error loss.
        
        Parameters:
        - y_true (Tensor): True target variable
        - y_pred (Tensor): Predicted target variable
        
        Returns:
        - ModuleOutput: the mean squared error loss as a ModuleOutput containing a single tensor
        """
        
        # If from_sequence is True, reshape the tensors for sequence-to-sequence loss
        if self.from_sequence:
            # Get the features size from the predictions
            features_size = y_pred.shape[-1]
            
            # Reshape predictions from (B, S, F) to (B*S, F)
            y_pred = y_pred.reshape((-1, features_size))
            
            # Reshape targets from (B, S, F) to (B*S, F) to match predictions
            y_true = y_true.reshape((-1, features_size))
        
        # Compute and return the loss
        return ModuleOutput(((y_true - y_pred) ** 2).mean())