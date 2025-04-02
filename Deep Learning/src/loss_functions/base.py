from ..core import Tensor


class LossFn:    
    
    ### Magic methods ###

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute the loss.

        Parameters:
        - y_true (Tensor): True target variable
        - y_pred (Tensor): Predicted target variable

        Returns:
        - Tensor: the loss value as a tensor
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method '__call__' is not implemented.")
