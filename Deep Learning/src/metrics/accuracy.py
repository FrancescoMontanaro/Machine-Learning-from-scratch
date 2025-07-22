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
    
    Raises:
    - ValueError: If tensors have incompatible shapes after processing
    """
    
    # Ensure the data is in numpy format
    y_true_data = y_true.to_numpy().copy()
    y_pred_data = y_pred.to_numpy().copy()
    
    # Handle multi-dimensional predictions (e.g., one-hot encoded or logits)
    if y_pred_data.ndim > 1:
        # Multi-class case (N, C)
        if y_pred_data.shape[-1] > 1:
            # Convert one-hot encoded predictions to class indices
            y_pred_data = np.argmax(y_pred_data, axis=-1)
        # Binary case with shape (N, 1)
        else: 
            # Squeeze to convert (N, 1) to (N,)
            y_pred_data = y_pred_data.squeeze()
    
    # Handle multi-dimensional true labels (e.g., one-hot encoded)
    if y_true_data.ndim > 1:
        # Multi-class case (N, C)
        if y_true_data.shape[-1] > 1:
            # Convert one-hot encoded labels to class indices
            y_true_data = np.argmax(y_true_data, axis=-1)
        # Binary case with shape (N, 1)
        else:
            # Squeeze to convert (N, 1) to (N,)
            y_true_data = y_true_data.squeeze()
    
    # Ensure both arrays are now 1D
    y_true_flat = y_true_data.ravel()
    y_pred_flat = y_pred_data.ravel()
    
    # Check for dimension mismatch after processing
    if len(y_true_flat) != len(y_pred_flat):
        # If the lengths do not match, raise an error
        raise ValueError(
            f"Shape mismatch after processing: y_true has {len(y_true_flat)} elements, "
            f"y_pred has {len(y_pred_flat)} elements. "
            f"Original shapes were y_true: {y_true_data.shape}, y_pred: {y_pred_data.shape}"
        )
    
    # Handle edge case of empty arrays
    if len(y_true_flat) == 0:
        return Tensor(0.0, requires_grad=False)
    
    # Compute and return the accuracy as a tensor
    return Tensor(np.mean(y_true_flat == y_pred_flat), requires_grad=False)