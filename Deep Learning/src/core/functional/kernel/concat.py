import numpy as np


def _normalize_axis(axis: int, rank: int) -> int:
    """
    Normalize an axis value into [0, rank).
    """

    # Validate axis boundaries before normalizing negative values
    if axis < -rank or axis >= rank:
        raise ValueError("axis out of range")

    # Normalize negative axis
    return axis + rank if axis < 0 else axis


def _validate_concat_shapes(ts_list: list[np.ndarray], axis: int) -> None:
    """
    Validate that all tensors are compatible for concatenation.
    """

    # Reference shape from the first tensor
    ref_shape = ts_list[0].shape
    ref_rank = len(ref_shape)

    # Validate rank and non-concatenated dimensions for all tensors
    for i, tensor in enumerate(ts_list[1:], 1):
        if tensor.ndim != ref_rank:
            raise ValueError(
                f"All tensors must have the same rank. Tensor 0 rank is {ref_rank}, "
                f"tensor {i} rank is {tensor.ndim}"
            )

        for dim in range(ref_rank):
            if dim == axis:
                continue
            if tensor.shape[dim] != ref_shape[dim]:
                raise ValueError(
                    f"All tensors must match in non-concatenated dimensions. "
                    f"Tensor 0 shape is {ref_shape}, tensor {i} shape is {tensor.shape}"
                )


def concat_forward(ts_list: list[np.ndarray], axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate tensors along a given axis and return:
    - concatenated output
    - metadata for backward: [axis, 0, off_1, ..., off_n]
      where off_k are cumulative offsets along the concatenation axis.
    """

    # Validate non-empty input
    if not ts_list:
        raise ValueError("ts_list cannot be empty")

    # Normalize and validate axis
    rank = ts_list[0].ndim
    axis = _normalize_axis(axis, rank)

    # Ensure dtype consistency (use first tensor dtype as canonical)
    first_dtype = ts_list[0].dtype
    if not all(t.dtype == first_dtype for t in ts_list):
        ts_list = [t.astype(first_dtype) for t in ts_list]

    # Validate shape compatibility
    _validate_concat_shapes(ts_list, axis)

    # Perform true axis-wise concatenation
    out = np.concatenate(ts_list, axis=axis)

    # Build backward metadata:
    # axis_offsets[0] = axis
    # axis_offsets[1:] = cumulative offsets along axis
    n = len(ts_list)
    axis_offsets = np.empty(n + 2, dtype=np.int64)
    axis_offsets[0] = axis
    axis_offsets[1] = 0
    for i, tensor in enumerate(ts_list):
        axis_offsets[i + 2] = axis_offsets[i + 1] + tensor.shape[axis]

    # Return output and metadata
    return out, axis_offsets


def concat_backward(out_grad: np.ndarray, out_buffer: np.ndarray, axis_offsets: np.ndarray, idx: int) -> None:
    """
    Backward pass for concat using axis-aware slicing.

    Parameters:
    - out_grad (np.ndarray): Gradient of concatenated output.
    - out_buffer (np.ndarray): Gradient buffer for tensor at position idx.
    - axis_offsets (np.ndarray): [axis, 0, off_1, ..., off_n].
    - idx (int): Index of tensor in original concatenation list.
    """

    # Validate metadata size and requested index
    if axis_offsets.size < 3:
        raise ValueError("Invalid concat metadata: axis_offsets too small")
    if idx < 0 or idx + 2 >= axis_offsets.size:
        raise ValueError("Invalid concat metadata: idx out of range")

    # Recover axis and segment bounds for this tensor
    axis = int(axis_offsets[0])
    start = int(axis_offsets[idx + 1])
    end = int(axis_offsets[idx + 2])

    # Build expected gradient shape for this slice
    expected_shape = list(out_grad.shape)
    expected_shape[axis] = end - start
    if tuple(expected_shape) != out_buffer.shape:
        raise ValueError(
            f"Concat backward shape mismatch. Expected {tuple(expected_shape)}, got {out_buffer.shape}"
        )

    # Slice the gradient segment for this tensor and copy into output buffer
    grad_slice = [slice(None)] * out_grad.ndim
    grad_slice[axis] = slice(start, end)
    np.copyto(out_buffer, out_grad[tuple(grad_slice)])