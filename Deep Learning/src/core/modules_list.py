from typing import Type, Generic, TypeVar, Union, TYPE_CHECKING, cast, overload
from .utils.types_registry import get_module_class

if TYPE_CHECKING: from .module import Module

# Define a type variable T that is bound to the Module class
T = TypeVar('T', bound='Module')


class ModuleList(Generic[T]):
    
    def __init__(self, modules: list[T]) -> None:
        """
        Class constructor.
        
        Parameters:
        - modules (list[Module]): List of modules to be stored in the ModuleList.
        
        Raises:
        - TypeError: If any of the elements in the list are not of type Module.
        """
        
        # Get the Module class from the registry
        Module = cast(Type['Module'], get_module_class())
        
        # Check if all elements in the list are of type Module
        if not all(isinstance(module, Module) for module in modules):
            raise TypeError("All elements in the list must be of type Module.")
        
        # Store the modules
        self.modules: list[T] = modules
        
    
    def __iter__(self):
        """
        Returns an iterator over the modules in the ModuleList.
        
        Returns:
        - iterator: An iterator over the modules in the ModuleList.
        """
        
        # Iterate over the modules
        return iter(self.modules)
    
    
    @overload
    def __getitem__(self, idx: int) -> T: 
        """
        Returns the module at the specified index.
        
        Parameters:
        - idx (int): The index of the module to be returned.
        
        Returns:
        - Module: The module at the specified index.
        """
        
        # Just to satisfy the overload
        ...
    
    
    @overload
    def __getitem__(self, idx: slice) -> list[T]: 
        """
        Returns a list of modules at the specified slice.
        
        Parameters:
        - idx (slice): The slice of the modules to be returned.
        
        Returns:
        - list[Module]: A list of modules at the specified slice.
        """
        
        # Just to satisfy the overload
        ...
    
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[T, list[T]]:
        """
        Returns the module at the specified index or a list of modules at the specified slice.
        
        Parameters:
        - idx (Union[int, slice]): The index or slice of the modules to be returned.
        
        Returns:
        - Union[Module, list[Module]]: The module at the specified index or a list of modules at the specified slice.
        """
        
        # Check if the index is an integer or a slice
        if isinstance(idx, slice):
            # Return a list of modules at the specified slice
            return self.modules[idx]
        
        # Return the module at the specified index
        return self.modules[idx]
    
    
    def __len__(self) -> int:
        """
        Returns the number of modules in the ModuleList.
        
        Returns:
        - int: The number of modules in the ModuleList.
        """
        
        # Return the length of the modules list
        return len(self.modules)