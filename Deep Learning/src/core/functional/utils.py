import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING: from ..tensor import Tensor


def accumulate_gradient(x: 'Tensor', grad: np.ndarray) -> None:
    """
    Accumulates gradients for a tensor.

    Parameters:
    - x (Tensor): Input tensor.
    - grad (Tensor): Gradient tensor.
    """
    
    # Accumulate the gradient
    x.grad = x.grad + grad if x.grad is not None else grad