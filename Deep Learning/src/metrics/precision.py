import numpy as np


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the precision of the model.

    Parameters:
    - y_true (np.ndarray): True target variable
    - y_pred (np.ndarray): Predicted target variable

    Returns:
    - float: Precision of the model
    """
    
    # Compute the precision
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    # Return the precision
    return tp / (tp + fp)