import numpy as np


def clip_forward(x_flat: np.ndarray, minv: float, maxv: float, out_flat: np.ndarray) -> None:
    """
    Clips the values in a flattened tensor to a specified range.
    
    Parameters:
    - x_flat (np.ndarray): Flattened input tensor.
    - minv (float): Minimum value for clipping.
    - maxv (float): Maximum value for clipping.
    - out_flat (np.ndarray): Flattened output tensor.
    """
    
    # Use numpy's optimized clip function
    np.clip(x_flat, minv, maxv, out=out_flat)


def clip_gradient(x_flat: np.ndarray, out_grad_flat: np.ndarray, x_grad_flat: np.ndarray, minv: float, maxv: float) -> None:
    """
    Computes the gradient of the clipping operation with respect to the input tensor.
    
    Parameters:
    - x_flat (np.ndarray): Flattened input tensor.
    - out_grad_flat (np.ndarray): Gradient of the output tensor.
    - x_grad_flat (np.ndarray): Gradient of the flattened input tensor.
    - minv (float): Minimum value for clipping.
    - maxv (float): Maximum value for clipping.
    """
    
    # Create mask for values within clipping range (vectorized)
    mask = (x_flat >= minv) & (x_flat <= maxv)
    
    # Propagate gradient only where values were not clipped
    x_grad_flat += out_grad_flat * mask