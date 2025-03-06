import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the mean absolute error of the model.

    Parameters:
    - y_true (np.ndarray): True target variable
    - y_pred (np.ndarray): Predicted target variable

    Returns:
    - float: Mean absolute error of the model
    """
    
    # Compute the mean absolute error
    return float(np.mean(np.abs(y_true - y_pred)))