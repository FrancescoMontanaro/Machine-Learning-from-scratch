import warnings
import numpy as np
from numba import njit, prange

# Disable warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
            

def matmul_forward(a_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication kernel for forward pass.
    
    Parameters:
    - a_data (np.ndarray): First input array.
    - b_data (np.ndarray): Second input array.
    
    Returns:
    - np.ndarray: Output array.
    """
    
    # Return the result of matrix multiplication
    return np.matmul(a_data, b_data)


@njit(parallel=True, fastmath=True)
def matmul_backward_a(out_grad: np.ndarray, out_buffer: np.ndarray, b_data: np.ndarray) -> None:
    """
    Matrix multiplication kernel for backward pass with respect to the first input.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output.
    - out_buffer (np.ndarray): Buffer to accumulate gradients.
    - b_data (np.ndarray): Second input array.
    """
    
    # Get the dimensions of the input arrays
    m, n = out_grad.shape[-2], out_grad.shape[-1]
    k = b_data.shape[-2]
    
    # Extract the batch size from the output gradient
    batch_size = out_grad.size // (m * n)
    
    # Broadcast the input arrays to the output shape
    OG = np.ascontiguousarray(np.broadcast_to(out_grad, (batch_size, m, n)))
    BT = np.ascontiguousarray(np.broadcast_to(np.swapaxes(b_data, -1, -2), (batch_size, n, k)))
    GA = np.ascontiguousarray(out_buffer).reshape(batch_size, m, k)

    # Iterate over the batches and perform matrix multiplication
    for i in prange(batch_size):
        # Perform matrix multiplication for each batch
        GA[i] += OG[i].dot(BT[i])


@njit(parallel=True, fastmath=True)
def matmul_backward_b(out_grad: np.ndarray, out_buffer: np.ndarray, a_data: np.ndarray) -> None:
    """
    Matrix multiplication kernel for backward pass with respect to the second input.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output.
    - out_buffer (np.ndarray): Buffer to accumulate gradients.
    - a_data (np.ndarray): First input array.
    """
    
    # Get the dimensions of the input arrays
    head = np.broadcast_shapes(a_data.shape[:-2], out_grad.shape[:-2])
    m, k = a_data.shape[-2], a_data.shape[-1]
    n = out_grad.shape[-1]

    # Extract the batch size from the output gradient
    batch_size = out_grad.size // (m * n)
    
    # Broadcast the input arrays to the output shape
    AT = np.ascontiguousarray(np.broadcast_to(np.swapaxes(a_data, -1, -2), head + (k, m))).reshape(batch_size, k, m)
    OG = np.ascontiguousarray(np.broadcast_to(out_grad, head + (m, n))).reshape(batch_size, m, n)
    GB = np.ascontiguousarray(out_buffer).reshape(batch_size, k, n)

    # Iterate over the batches and perform matrix multiplication
    for i in prange(batch_size):
        # Perform matrix multiplication for each batch
        GB[i] += AT[i].dot(OG[i])