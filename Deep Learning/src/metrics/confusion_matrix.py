import numpy as np


def confusion_matrix(num_classes: int, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the confusion matrix of the model.

    Parameters:
    - num_classes (int): Number of classes
    - y_true (np.ndarray): True target variable
    - y_pred (np.ndarray): Predicted target variable

    Returns:
    - np.ndarray: Confusion matrix
    """
    
    # Arrays must be 1D
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=-1)
        
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    
    # Compute the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    # Fill the confusion matrix
    for i in range(len(y_true)):
        # Increment the confusion matrix
        confusion_matrix[y_true[i], y_pred[i]] += 1
        
    return confusion_matrix