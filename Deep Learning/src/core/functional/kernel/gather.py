from numba import njit, prange


@njit(parallel=True)
def gather_forward(x_flat, idx_flat, stride, out_flat) -> None:
    """
    Gathers elements from a flattened tensor based on indices.
    
    Parameters:
    - x_flat (np.ndarray): Flattened input tensor.
    - idx_flat (np.ndarray): Indices to gather from the input tensor.
    - stride (int): Number of elements per "row" in the original linearization.
    - out_flat (np.ndarray): Flattened output tensor.
    """
    
    # Iterate over the flattened tensor
    for i in prange(out_flat.size):
        # Add the stride to the index to get the correct position in the original tensor
        idx_flat[i] = idx_flat[i] * stride
        
        # Gather the element from the input tensor
        out_flat[i] = x_flat[idx_flat[i]]


@njit(parallel=True)
def gather_gradient(idx_flat, out_grad_flat, x_grad_flat) -> None:
    """
    Computes the gradient of the gather operation with respect to the input tensor.
    
    Parameters:
    - idx_flat (np.ndarray): Indices used in the gather operation.
    - out_grad_flat (np.ndarray): Gradient of the output tensor.
    - x_grad_flat (np.ndarray): Gradient of the flattened input tensor.
    """
    
    # Iterate over the flattened tensor
    for i in prange(out_grad_flat.size):
        # Add the output gradient to the input gradient at the gathered index
        x_grad_flat[idx_flat[i]] += out_grad_flat[i]