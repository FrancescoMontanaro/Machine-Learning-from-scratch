import numpy as np
        
        
def pow_forward(t: np.ndarray, power: float) -> np.ndarray:
    """
    Forward pass for element-wise power operation.
    
    Parameters:
    - t (np.ndarray): The input array to be raised to the power
    - power (float): Power to raise the elements of the first array
    
    Returns:
    - np.ndarray: Output array with the result of raising each element to the power
    """
    
    # Optimize for common integer powers using multiplication
    # This is much faster than np.power for small integer exponents
    if power == 2:
        return t * t
    elif power == 3:
        return t * t * t
    elif power == 4:
        t2 = t * t
        return t2 * t2
    elif power == 0:
        return np.ones_like(t)
    elif power == 1:
        return t.copy()
    elif power == -1:
        return 1.0 / t
    elif power == 0.5:
        return np.sqrt(t)
    else:
        # Fall back to np.power for other cases
        return np.power(t, power)


def pow_gradient(out_grad: np.ndarray, t: np.ndarray, power: float) -> np.ndarray:
    """
    Computes the gradients for the inputs of the elementwise raise to the power operation.
    
    Parameters:
    - out_grad (np.ndarray): Gradient of the output
    - t (np.ndarray): The input array to be raised to the power
    - power (float): Power to raise the elements of the first array
    
    Returns:
    - np.ndarray: Gradient of the input array
    """
    
    # Optimize gradient computation for common integer powers
    if power == 2:
        return (2.0 * t) * out_grad
    elif power == 3:
        return (3.0 * t * t) * out_grad
    elif power == 4:
        t2 = t * t
        return (4.0 * t2 * t) * out_grad
    elif power == 0:
        return np.zeros_like(t)
    elif power == 1:
        return out_grad.copy()
    elif power == -1:
        return (-1.0 / (t * t)) * out_grad
    elif power == 0.5:
        return (0.5 / np.sqrt(t)) * out_grad
    else:
        # Fall back to general formula
        return power * np.power(t, power - 1) * out_grad