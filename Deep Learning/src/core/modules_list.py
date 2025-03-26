from typing import Type, TYPE_CHECKING, cast

if TYPE_CHECKING: from .module import Module
from .utils.types_registry import get_module_class


class ModuleList:
    
    ### Magic methods ###
    
    def __init__(self, modules: list['Module']) -> None:
        """
        Constructor for the ModuleList class
        
        Parameters:
        - modules (list[Module]): List of modules to store
        
        Raises:
        - TypeError: If the input is not a list of modules
        """
        
        # Get the module class
        Module = cast(Type['Module'], get_module_class())
        
        # Check if the input is a list of modules
        if not all(isinstance(module, Module) for module in modules):
            raise TypeError("All elements in the list must be of type Module!")
        
        # Store the list of modules
        self.modules = modules
        
        
    def __iter__(self):
        """
        Method to iterate over the list of modules
        """
        
        # Return the iterator over the list of modules
        return iter(self.modules)
    
    
    def __getitem__(self, idx: int) -> 'Module':
        """
        Method to get a module from the list
        
        Parameters:
        - idx (int): Index of the module to retrieve
        
        Returns:
        - Module: The module at the given index
        """
        
        # Return the module at the given index
        return self.modules[idx]