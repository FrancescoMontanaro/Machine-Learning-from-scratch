_Tensor = None

def get_tensor_class() -> type:
    """
    Lazily import and return the Tensor class to avoid circular imports.
    
    Returns:
    - Tensor class (type)
    """
    
    # Access the global variable
    global _Tensor
    
    # If the Tensor class has not been imported, import it
    if _Tensor is None:
        # Import the Tensor class
        from .tensor import Tensor
        
        # Store the Tensor class
        _Tensor = Tensor
        
    # Return the Tensor class
    return _Tensor