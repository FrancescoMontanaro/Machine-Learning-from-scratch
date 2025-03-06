import numpy as np

from .recall import recall
from .precision import precision


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the F1 score of the model.

    Parameters:
    - y_true (np.ndarray): True target variable
    - y_pred (np.ndarray): Predicted target variable

    Returns:
    - float: F1 score of the model
    """
    
    # Compute the precision and recall
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    
    # Compute the F1 score
    return 2 * (prec * rec) / (prec + rec)