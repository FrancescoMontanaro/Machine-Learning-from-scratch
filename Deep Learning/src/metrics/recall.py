import numpy as np


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the recall of the model.

    Parameters:
    - y_true (np.ndarray): True target variable
    - y_pred (np.ndarray): Predicted target variable

    Returns:
    - float: Recall of the model
    """
    
    # Compute the recall
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Return the recall
    return tp / (tp + fn)