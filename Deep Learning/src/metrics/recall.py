import numpy as np

from ..core import Tensor


def recall(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Compute the recall of the model.

    Parameters:
    - y_true (Tensor): True target variable
    - y_pred (Tensor): Predicted target variable

    Returns:
    - Tensor: Recall of the model
    """
    
    # Compute the recall
    tp = np.sum((y_true.to_numpy() == 1) & (y_pred.to_numpy() == 1))
    fn = np.sum((y_true.to_numpy() == 1) & (y_pred.to_numpy() == 0))
    
    # Compute and return the recall as a tensor
    return Tensor((tp / (tp + fn)), requires_grad=False)