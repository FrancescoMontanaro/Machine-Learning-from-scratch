import numpy as np

from ..core import Tensor


def confusion_matrix(num_classes: int, y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Compute the confusion matrix of the model.

    Parameters:
    - num_classes (int): Number of classes
    - y_true (Tensor): True target variable
    - y_pred (Tensor): Predicted target variable

    Returns:
    - Tensor: Confusion matrix
    """
    
    # Extract the data
    y_true_data = y_true.data.copy()
    y_pred_data = y_pred.data.copy()
    
    # Arrays must be 1D
    if y_true_data.ndim > 1:
        y_true_data = np.argmax(y_true_data, axis=-1)
        
    if y_pred_data.ndim > 1:
        y_pred_data = np.argmax(y_pred_data, axis=-1)
    
    # Compute the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    # Fill the confusion matrix
    for i in range(len(y_true_data)):
        # Increment the confusion matrix
        confusion_matrix[y_true_data[i], y_true_data[i]] += 1
        
    # Return the confusion matrix as a tensor
    return Tensor(confusion_matrix, requires_grad=False)