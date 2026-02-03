import numpy as np


def unsqueeze_gradient(out_grad_flat: np.ndarray, x_grad_flat: np.ndarray) -> None:
    """
    Computes the gradient of the unsqueeze operation.
    
    Parameters:
    - out_grad_flat (np.ndarray): Gradient of the output tensor, flattened.
    - x_grad_flat (np.ndarray): Gradient of the input tensor, flattened.
    """
    
    # Vectorized addition - much faster than element-wise loop
    x_grad_flat += out_grad_flat