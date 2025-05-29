import numpy as np

from ..core import Tensor


def mean_absolute_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Compute the mean absolute error of the model.

    Parameters:
    - y_true (Tensor): True target variable
    - y_pred (Tensor): Predicted target variable

    Returns:
    - Tensor: Mean absolute error of the model
    """
    
    # Compute and return the mean absolute error as a tensor
    return Tensor(float(np.mean(np.abs(y_true.to_numpy() - y_pred.to_numpy()))), requires_grad=False)