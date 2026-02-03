import numpy as np


def pad_forward(x: np.ndarray, pad_top: int, pad_bottom: int, pad_left: int, pad_right: int, out: np.ndarray) -> None:
    """
    2D Padding forward pass.
    
    Parameters:
    - x (np.ndarray): Input tensor of shape (batch_size, height, width, channels)
    - pad_top (int): Number of rows to pad at the top
    - pad_bottom (int): Number of rows to pad at the bottom
    - pad_left (int): Number of columns to pad at the left
    - pad_right (int): Number of columns to pad at the right
    - out (np.ndarray): Output tensor of shape (batch_size, height + pad_top + pad_bottom, width + pad_left + pad_right, channels)
    """
    
    # Extract dimensions
    _, height, width, _ = x.shape
    
    # Zero out the output (for padding regions)
    out.fill(0.0)
    
    # Compute end indices for the non-padded region
    h_end = pad_top + height
    w_end = pad_left + width
    
    # Copy input to the appropriate region using slicing (vectorized)
    out[:, pad_top:h_end, pad_left:w_end, :] = x


def pad_gradient(out_grad: np.ndarray, pad_top: int, pad_bottom: int, pad_left: int, pad_right: int, x_grad: np.ndarray) -> None:
    """
    2D Padding gradient
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output tensor of shape (batch_size, output_height, output_width, channels)
    - pad_top (int): Number of rows padded at the top
    - pad_bottom (int): Number of rows padded at the bottom
    - pad_left (int): Number of columns padded at the left
    - pad_right (int): Number of columns padded at the right
    - x_grad (np.ndarray): Gradient of the input tensor of shape (batch_size, height, width, channels)
    """
    
    # Extract dimensions
    height, width = x_grad.shape[1], x_grad.shape[2]
    
    # Compute end indices for the non-padded region  
    h_end = pad_top + height
    w_end = pad_left + width
    
    # Extract gradient from the non-padded region and accumulate (vectorized)
    x_grad += out_grad[:, pad_top:h_end, pad_left:w_end, :]