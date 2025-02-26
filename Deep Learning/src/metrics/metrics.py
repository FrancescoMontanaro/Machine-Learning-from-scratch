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