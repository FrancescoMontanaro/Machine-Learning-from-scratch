import numpy as np

from ..core import Tensor


def precision(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Compute the precision of the model.

    Parameters:
    - y_true (Tensor): True target variable
    - y_pred (Tensor): Predicted target variable

    Returns:
    - Tensor: Precision of the model
    """
    
    # Compute the precision
    tp = np.sum((y_true.to_numpy() == 1) & (y_pred.to_numpy() == 1))
    fp = np.sum((y_true.to_numpy() == 0) & (y_pred.to_numpy() == 1))
    
    # Compute and return the precision as a tensor
    return Tensor((tp / (tp + fp)), requires_grad=False)