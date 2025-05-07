import functools
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING: from ..tensor import Tensor; from ..module import Module


@functools.lru_cache(maxsize=1)
def get_tensor_class() -> Type['Tensor']:
    """
    Get the Tensor class from the module.
    This is a lazy import to avoid circular dependencies.
    
    Returns:
    - Type[Tensor]: The Tensor class.
    """
    
    # Lazy import to avoid circular dependencies
    from ..tensor import Tensor
    
    # Return the Tensor class
    return Tensor


@functools.lru_cache(maxsize=1)
def get_module_class() -> Type['Module']:
    """
    Get the Module class from the module.
    This is a lazy import to avoid circular dependencies.
    
    Returns:
    - Type[Module]: The Module class.
    """
    
    # Lazy import to avoid circular dependencies
    from ..module import Module
    
    # Return the Module class
    return Module