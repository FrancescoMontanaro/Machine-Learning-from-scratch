from numba import njit, prange


@njit(parallel=True)
def masked_fill_forward(x_flat, mask_flat, value, out_flat) -> None:
    """
    Applies a masked fill operation on a flattened tensor.
    
    Parameters:
    - x_flat (np.ndarray): Flattened input tensor.
    - mask_flat (np.ndarray): Flattened boolean mask tensor.
    - value (float): Value to fill in the masked positions.
    - out_flat (np.ndarray): Flattened output tensor.
    """
    
    # Iterate over the flattened tensor
    for i in prange(x_flat.size):
        # If the mask is True, keep the original value; otherwise, fill with the specified value
        out_flat[i] = x_flat[i] if mask_flat[i] else value


@njit(parallel=True)
def masked_fill_gradient(mask_flat, out_grad_flat, x_grad_flat) -> None:
    """
    Computes the gradient of the masked fill operation with respect to the input tensor.
    
    Parameters:
    - mask_flat (np.ndarray): Flattened boolean mask tensor.
    - out_grad_flat (np.ndarray): Gradient of the output tensor.
    - x_grad_flat (np.ndarray): Gradient of the flattened input tensor.
    """
    
    # Iterate over the flattened tensor
    for i in prange(mask_flat.size):
        # If the mask is True, propagate the gradient; otherwise, add the output gradient
        if not mask_flat[i]:
            # Add the output gradient to the input gradient
            x_grad_flat[i] += out_grad_flat[i]