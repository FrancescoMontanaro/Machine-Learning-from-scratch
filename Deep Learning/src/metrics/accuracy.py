import numpy as np

from ..core import Tensor


def accuracy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Compute the accuracy of the model.

    Parameters:
    - y_true (Tensor): True target variable
    - y_pred (Tensor): Predicted target variable

    Returns:
    - Tensor: Accuracy tensor
    """
    
    # Ensure the data is in numpy format
    y_true_data = y_true.to_numpy().copy()
    y_pred_data = y_pred.to_numpy().copy()
    
    # If the array is multi-dimensional, take the argmax
    if y_pred_data.ndim > 1:
        y_pred_data = np.argmax(y_pred_data, axis=-1)
        
    if y_true.to_numpy().ndim == y_pred.to_numpy().ndim:
        y_true_data = np.argmax(y_true_data, axis=-1)
        
    # Ensure the arrays are 1D
    y_true_flat = y_true_data.ravel()
    y_pred_flat = y_pred_data.ravel()
    
    # Compute and return the accuracy as a tensor
    return Tensor(np.mean(y_true_flat == y_pred_flat), requires_grad=False)