import numpy as np
from numba import njit, prange
from typing import Union, Optional, Tuple


# ? ############################## ? #
# ? ####### FORWARD PASS ######### ? #
# ? ############################## ? #

######################
### Rank 1 kernels ###
######################

@njit(parallel=True, fastmath=True)
def _sum_all_1d(x: np.ndarray) -> np.ndarray:
    acc = x.dtype.type(0)
    for i in prange(x.size):
        acc += x[i]
    return acc

# --- keepdims wrappers --- #

@njit(fastmath=True)
def _sum_all_1d_kd_true(x: np.ndarray) -> np.ndarray:
    return np.reshape(_sum_all_1d(x), (1,))

@njit(fastmath=True)
def _sum_all_1d_kd_false(x: np.ndarray) -> np.ndarray:
    return _sum_all_1d(x)


######################
### Rank 2 kernels ###
######################

@njit(parallel=True, fastmath=True)
def _sum_axis0_2d(x: np.ndarray) -> np.ndarray:
    m, n = x.shape
    out = np.zeros(n, dtype=x.dtype)
    for j in prange(n):
        s = x.dtype.type(0)
        for i in range(m):
            s += x[i, j]
        out[j] = s
    return out

@njit(parallel=True, fastmath=True)
def _sum_axis1_2d(x: np.ndarray) -> np.ndarray:
    m, n = x.shape
    out = np.zeros(m, dtype=x.dtype)
    for i in prange(m):
        s = x.dtype.type(0)
        for j in range(n):
            s += x[i, j]
        out[i] = s
    return out

@njit(parallel=True, fastmath=True)
def _sum_all_2d(x: np.ndarray) -> np.ndarray:
    m, n = x.shape
    acc = x.dtype.type(0)
    for i in prange(m):
        local = x.dtype.type(0)
        for j in range(n):
            local += x[i, j]
        acc += local
    return acc

# --- keepdims wrappers --- #

@njit(fastmath=True)
def _sum_axis0_2d_kd_true(x: np.ndarray) -> np.ndarray:
    return np.reshape(_sum_axis0_2d(x), (1, x.shape[1]))

@njit(fastmath=True)
def _sum_axis0_2d_kd_false(x: np.ndarray) -> np.ndarray:
    return _sum_axis0_2d(x)

@njit(fastmath=True)
def _sum_axis1_2d_kd_true(x: np.ndarray) -> np.ndarray:
    return np.reshape(_sum_axis1_2d(x), (x.shape[0], 1))

@njit(fastmath=True)
def _sum_axis1_2d_kd_false(x: np.ndarray) -> np.ndarray:
    return _sum_axis1_2d(x)

@njit(fastmath=True)
def _sum_all_2d_kd_true(x: np.ndarray) -> np.ndarray:
    return np.reshape(_sum_all_2d(x), (1, 1))

@njit(fastmath=True)
def _sum_all_2d_kd_false(x: np.ndarray) -> np.ndarray:
    return _sum_all_2d(x)


######################
### Rank 3 kernels ###
######################

@njit(parallel=True, fastmath=True)
def _sum_axis0_3d(x: np.ndarray) -> np.ndarray:
    d0, d1, d2 = x.shape
    out = np.zeros((d1, d2), dtype=x.dtype)
    for j in prange(d1):
        for k in range(d2):
            s = x.dtype.type(0)
            for i in range(d0):
                s += x[i, j, k]
            out[j, k] = s
    return out

@njit(parallel=True, fastmath=True)
def _sum_axis1_3d(x: np.ndarray) -> np.ndarray:
    d0, d1, d2 = x.shape
    out = np.zeros((d0, d2), dtype=x.dtype)
    for i in prange(d0):
        for k in range(d2):
            s = x.dtype.type(0)
            for j in range(d1):
                s += x[i, j, k]
            out[i, k] = s
    return out

@njit(parallel=True, fastmath=True)
def _sum_axis2_3d(x: np.ndarray) -> np.ndarray:
    d0, d1, d2 = x.shape
    out = np.zeros((d0, d1), dtype=x.dtype)
    for i in prange(d0):
        for j in range(d1):
            s = x.dtype.type(0)
            for k in range(d2):
                s += x[i, j, k]
            out[i, j] = s
    return out

@njit(parallel=True, fastmath=True)
def _sum_all_3d(x: np.ndarray) -> np.ndarray:
    d0, d1, d2 = x.shape
    acc = x.dtype.type(0)
    for i in prange(d0):
        local_i = x.dtype.type(0)
        for j in range(d1):
            local_j = x.dtype.type(0)
            for k in range(d2):
                local_j += x[i, j, k]
            local_i += local_j
        acc += local_i
    return acc

# --- keepdims wrappers --- #

@njit(fastmath=True)
def _sum_axis0_3d_kd_true(x: np.ndarray) -> np.ndarray:
    d1, d2 = x.shape[1:]
    return np.reshape(_sum_axis0_3d(x), (1, d1, d2))

@njit(fastmath=True)
def _sum_axis0_3d_kd_false(x: np.ndarray) -> np.ndarray:
    return _sum_axis0_3d(x)

@njit(fastmath=True)
def _sum_axis1_3d_kd_true(x: np.ndarray) -> np.ndarray:
    d0, d2 = x.shape[0], x.shape[2]
    return np.reshape(_sum_axis1_3d(x), (d0, 1, d2))

@njit(fastmath=True)
def _sum_axis1_3d_kd_false(x: np.ndarray) -> np.ndarray:
    return _sum_axis1_3d(x)

@njit(fastmath=True)
def _sum_axis2_3d_kd_true(x: np.ndarray) -> np.ndarray:
    d0, d1 = x.shape[:2]
    return np.reshape(_sum_axis2_3d(x), (d0, d1, 1))

@njit(fastmath=True)
def _sum_axis2_3d_kd_false(x: np.ndarray) -> np.ndarray:
    return _sum_axis2_3d(x)

@njit(fastmath=True)
def _sum_all_3d_kd_true(x: np.ndarray) -> np.ndarray:
    return np.reshape(_sum_all_3d(x), (1, 1, 1))

@njit(fastmath=True)
def _sum_all_3d_kd_false(x: np.ndarray) -> np.ndarray:
    return _sum_all_3d(x)


######################
### Rank 4 kernels ###
######################

@njit(parallel=True, fastmath=True)
def _sum_axis0_4d(x: np.ndarray) -> np.ndarray:
    d0, d1, d2, d3 = x.shape
    out = np.zeros((d1, d2, d3), dtype=x.dtype)
    for j in prange(d1):
        for k in range(d2):
            for l in range(d3):
                s = x.dtype.type(0)
                for i in range(d0):
                    s += x[i, j, k, l]
                out[j, k, l] = s
    return out

@njit(parallel=True, fastmath=True)
def _sum_axis1_4d(x: np.ndarray) -> np.ndarray:
    d0, d1, d2, d3 = x.shape
    out = np.zeros((d0, d2, d3), dtype=x.dtype)
    for i in prange(d0):
        for k in range(d2):
            for l in range(d3):
                s = x.dtype.type(0)
                for j in range(d1):
                    s += x[i, j, k, l]
                out[i, k, l] = s
    return out

@njit(parallel=True, fastmath=True)
def _sum_axis2_4d(x: np.ndarray) -> np.ndarray:
    d0, d1, d2, d3 = x.shape
    out = np.zeros((d0, d1, d3), dtype=x.dtype)
    for i in prange(d0):
        for j in range(d1):
            for l in range(d3):
                s = x.dtype.type(0)
                for k in range(d2):
                    s += x[i, j, k, l]
                out[i, j, l] = s
    return out

@njit(parallel=True, fastmath=True)
def _sum_axis3_4d(x: np.ndarray) -> np.ndarray:
    d0, d1, d2, d3 = x.shape
    out = np.zeros((d0, d1, d2), dtype=x.dtype)
    for i in prange(d0):
        for j in range(d1):
            for k in range(d2):
                s = x.dtype.type(0)
                for l in range(d3):
                    s += x[i, j, k, l]
                out[i, j, k] = s
    return out

@njit(parallel=True, fastmath=True)
def _sum_all_4d(x: np.ndarray) -> np.ndarray:
    d0, d1, d2, d3 = x.shape
    acc = x.dtype.type(0)
    for i in prange(d0):
        local_i = x.dtype.type(0)
        for j in range(d1):
            local_j = x.dtype.type(0)
            for k in range(d2):
                local_k = x.dtype.type(0)
                for l in range(d3):
                    local_k += x[i, j, k, l]
                local_j += local_k
            local_i += local_j
        acc += local_i
    return acc

# --- keepdims wrappers --- #

@njit(fastmath=True)
def _sum_axis0_4d_kd_true(x: np.ndarray) -> np.ndarray:
    d1, d2, d3 = x.shape[1:]
    return np.reshape(_sum_axis0_4d(x), (1, d1, d2, d3))

@njit(fastmath=True)
def _sum_axis0_4d_kd_false(x: np.ndarray) -> np.ndarray:
    return _sum_axis0_4d(x)

@njit(fastmath=True)
def _sum_axis1_4d_kd_true(x: np.ndarray) -> np.ndarray:
    d0, d2, d3 = x.shape[0], x.shape[2], x.shape[3]
    return np.reshape(_sum_axis1_4d(x), (d0, 1, d2, d3))

@njit(fastmath=True)
def _sum_axis1_4d_kd_false(x: np.ndarray) -> np.ndarray:
    return _sum_axis1_4d(x)

@njit(fastmath=True)
def _sum_axis2_4d_kd_true(x: np.ndarray) -> np.ndarray:
    d0, d1, d3 = x.shape[0], x.shape[1], x.shape[3]
    return np.reshape(_sum_axis2_4d(x), (d0, d1, 1, d3))

@njit(fastmath=True)
def _sum_axis2_4d_kd_false(x: np.ndarray) -> np.ndarray:
    return _sum_axis2_4d(x)

@njit(fastmath=True)
def _sum_axis3_4d_kd_true(x: np.ndarray) -> np.ndarray:
    d0, d1, d2 = x.shape[:3]
    return np.reshape(_sum_axis3_4d(x), (d0, d1, d2, 1))

@njit(fastmath=True)
def _sum_axis3_4d_kd_false(x: np.ndarray) -> np.ndarray:
    return _sum_axis3_4d(x)

@njit(fastmath=True)
def _sum_all_4d_kd_true(x: np.ndarray) -> np.ndarray:
    return np.reshape(_sum_all_4d(x), (1, 1, 1, 1))

@njit(fastmath=True)
def _sum_all_4d_kd_false(x: np.ndarray) -> np.ndarray:
    return _sum_all_4d(x)


def _reduce_multi_axes(x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> np.ndarray:
    # Reduce the tensor along multiple axes using numpy's sum
    return np.sum(x, axis=axis, keepdims=keepdims)


def sum_forward(x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> np.ndarray:
    """
    Computes the sum of the elements along the specified axis or axes.
    
    Parameters:
    - x (np.ndarray): Input tensor.
    - axis (Optional[Union[int, Tuple[int]]]): Axis or axes to reduce
    - keepdims (bool): Whether to keep the dimensions of the output
    
    Returns:
    - np.ndarray: Output tensor with the sum of the elements along the specified axis or axes.
    """
    
    
    # If axis is None, return the sum of all elements
    if axis is None:
        # Return the sum of all elements in the tensor
        return (
            _sum_all_1d_kd_true(x) if x.ndim == 1 and keepdims else
            _sum_all_2d_kd_true(x) if x.ndim == 2 and keepdims else
            _sum_all_3d_kd_true(x) if x.ndim == 3 and keepdims else
            _sum_all_4d_kd_true(x) if x.ndim == 4 and keepdims else
            _sum_all_1d_kd_false(x) if x.ndim == 1 else
            _sum_all_2d_kd_false(x) if x.ndim == 2 else
            _sum_all_3d_kd_false(x) if x.ndim == 3 else
            _sum_all_4d_kd_false(x)
        )

    # If axis is an integer, convert it to a tuple
    if isinstance(axis, int):
        # Convert negative axis to positive
        if axis < 0:
            axis += x.ndim
            
        # Sum over the specified axis
        if x.ndim == 1:
            return _sum_all_1d_kd_true(x) if keepdims else _sum_all_1d_kd_false(x)
        elif x.ndim == 2:
            return (
                _sum_axis0_2d_kd_true(x) if axis == 0 and keepdims else
                _sum_axis1_2d_kd_true(x) if axis == 1 and keepdims else
                _sum_axis0_2d_kd_false(x) if axis == 0 else
                _sum_axis1_2d_kd_false(x)
            )
        elif x.ndim == 3:
            return (
                _sum_axis0_3d_kd_true(x) if axis == 0 and keepdims else
                _sum_axis1_3d_kd_true(x) if axis == 1 and keepdims else
                _sum_axis2_3d_kd_true(x) if axis == 2 and keepdims else
                _sum_axis0_3d_kd_false(x) if axis == 0 else
                _sum_axis1_3d_kd_false(x) if axis == 1 else
                _sum_axis2_3d_kd_false(x)
            )
                   
        elif x.ndim == 4:
            return (
                _sum_axis0_4d_kd_true(x) if axis == 0 and keepdims else
                _sum_axis1_4d_kd_true(x) if axis == 1 and keepdims else
                _sum_axis2_4d_kd_true(x) if axis == 2 and keepdims else
                _sum_axis3_4d_kd_true(x) if axis == 3 and keepdims else
                _sum_axis0_4d_kd_false(x) if axis == 0 else
                _sum_axis1_4d_kd_false(x) if axis == 1 else
                _sum_axis2_4d_kd_false(x) if axis == 2 else
                _sum_axis3_4d_kd_false(x)
            )

    # Reduce the tensor along multiple axes
    return _reduce_multi_axes(x, axis, keepdims)


# ? ############################## ? #
# ? ####### BACKWARD PASS ######## ? #
# ? ############################## ? #


def sum_backward(out_grad: np.ndarray, out_buffer: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> None:
    """
    Backward pass for the sum operation.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output
    - out_buffer (np.ndarray): Buffer to store the result
    - axis (Optional[Union[int, Tuple[int]]]): Axis or axes to reduce
    - keepdims (bool): Whether to keep the dimensions of the output
    """
    
    # Extract the rank of the output buffer
    ndim = len(out_buffer.shape)
    
    # Convert the axes to a tuple of integers
    if axis is None:
        axes = tuple(range(ndim))
    elif isinstance(axis, int):
        axes = tuple(sorted({a % ndim for a in (axis,)}))
    else:
        axes = tuple(sorted({a % ndim for a in axis}))

    # If keepdims is False, expand the dimensions of out_grad
    if not keepdims:
        # Iterate over the axes in reverse order
        for ax in axes:
            # Expand the dimensions of out_grad
            out_grad = np.expand_dims(out_grad, ax)

    # Broadcast out_grad to the shape of out_buffer and add it to out_buffer
    out_buffer += np.broadcast_to(out_grad, out_buffer.shape)