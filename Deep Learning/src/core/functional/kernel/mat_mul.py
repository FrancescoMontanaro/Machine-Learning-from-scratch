import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def matmul_forward(a_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication kernel for forward pass.
    
    Parameters:
    - a_data (np.ndarray): First input array.
    - b_data (np.ndarray): Second input array.
    
    Returns:
    - np.ndarray: Result of the matrix multiplication.
    """
    
    # Define the output shape of the matrix multiplication
    out_shape = np.broadcast_shapes(a_data.shape[:-2], b_data.shape[:-2]) + (a_data.shape[-2], b_data.shape[-1])
    
    # Create an empty output array with the appropriate shape
    out = np.empty(out_shape, dtype=a_data.dtype)

    # Get the dimensions of the input arrays
    m, k = a_data.shape[-2], a_data.shape[-1]
    n = b_data.shape[-1]
    
    # Broadcast the input arrays to the output shape
    a_view = np.broadcast_to(a_data, out_shape[:-2] + (m, k))
    b_view = np.broadcast_to(b_data, out_shape[:-2] + (k, n))

    # Flatten the output array for easier manipulation
    batch_size = out.size // (m * n)

    # Create contiguous views of the input arrays
    A = np.ascontiguousarray(a_view).reshape(batch_size, m, k)
    B = np.ascontiguousarray(b_view).reshape(batch_size, k, n)
    O = out.reshape(batch_size, m, n)

    # Iterate over the batches and perform matrix multiplication
    for i in prange(batch_size):
        # Perform matrix multiplication for each batch
        O[i] = A[i].dot(B[i])

    return out


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