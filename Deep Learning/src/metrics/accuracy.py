import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the accuracy of the model.

    Parameters:
    - y_true (np.ndarray): True target variable
    - y_pred (np.ndarray): Predicted target variable

    Returns:
    - float: Accuracy value
    """
    
    # If the array is multi-dimensional, take the argmax
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=-1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    
    # Compute the accuracy
    return np.mean(y_true == y_pred, axis=-1)