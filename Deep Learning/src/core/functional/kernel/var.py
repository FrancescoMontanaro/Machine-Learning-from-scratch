from numba import njit, prange


@njit(parallel=True)
def var_flat_forward(x_flat, out_scalar, ddof) -> None:
    """
    Computes the variance of a flattened tensor.
    
    Parameters:
    - x_flat (np.ndarray): Flattened input tensor.
    - out_scalar (np.ndarray): Output scalar to store the variance.
    - ddof (int): Delta degrees of freedom for variance calculation.
    """
    
    # Extract the number of elements in the flattened tensor
    n = x_flat.size
    
    # Initialize the sum to zero
    s = 0.0
    
    # iterate over the flattened tensor to compute the sum
    for i in prange(n): 
        s += x_flat[i]
        
    # Compute the mean
    m = s / n
    
    # Initialize the sum of squared differences to zero
    ss = 0.0
    
    # iterate over the flattened tensor to compute the sum of squares
    for i in prange(n):
        # compute the squared difference from the mean
        d = x_flat[i] - m
        
        # accumulate the squared difference
        ss += d*d
        
    # If ddof is not zero and n is greater than ddof, adjust the variance calculation
    if ddof != 0 and n > ddof:
        # Adjust the sum of squares by dividing by n - ddof
        out_scalar[0] = ss / (n - ddof)
    else:
        # Adjust the sum of squares by dividing by n
        out_scalar[0] = ss / n


@njit(parallel=True)
def var_flat_gradient(out_grad_scalar, x_flat, x_grad_flat, ddof) -> None:
    """
    Computes the gradient of the variance operation with respect to the input tensor.
    
    Parameters:
    - out_grad_scalar (np.ndarray): Gradient of the output scalar.
    - x_flat (np.ndarray): Flattened input tensor.
    - x_grad_flat (np.ndarray): Gradient of the flattened input tensor.
    - ddof (int): Delta degrees of freedom for variance calculation.
    """
    
    # Extract the number of elements in the flattened tensor
    n = x_flat.size
    
    # Initialize the sum to zero
    s = 0.0
    
    # iterate over the flattened tensor to compute the sum
    for i in prange(n): 
        s += x_flat[i]
        
    # Compute the mean
    m = s / n
    
    # Compute the inverse count for normalization
    inv = 1.0 / (n - ddof) if ddof != 0 and n > ddof else 1.0 / n
    
    # Initialize the sum of squared differences to zero
    vg = out_grad_scalar[0]
    
    # iterate over the flattened tensor to compute the sum of squares
    for i in prange(n):
        # compute the squared difference from the mean
        x_grad_flat[i] += 2 * (x_flat[i] - m) * inv * vg