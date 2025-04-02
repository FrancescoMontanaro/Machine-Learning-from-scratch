from typing import Generic, TypeVar, Union, overload

from . import Tensor

from .module import Module

# Define a type variable T that is bound to the Module class
T = TypeVar('T', bound=Module)


class ModuleList(Module, Generic[T]):
    
    ### Magic Methods ###
    
    def __init__(self, modules: list[Module], *args, **kwargs) -> None:
        """
        Class constructor.
        
        Parameters:
        - modules (list[Module]): List of modules to be stored in the ModuleList.
        
        Raises:
        - TypeError: If any of the elements in the list are not of type Module.
        """
        
        # Call the parent class constructor
        super().__init__(*args, **kwargs)
        
        # Check if all elements in the list are of type Module
        if not all(isinstance(module, Module) for module in modules):
            raise TypeError("All elements in the list must be of type Module.")
        
        # Iterate over the modules
        for i, module in enumerate(modules):
            # Set the attribute name to the index
            setattr(self, str(i), module)

    
    def __iter__(self):
        """
        Returns an iterator over the modules in the ModuleList.
        
        Returns:
        - iterator: An iterator over the modules in the ModuleList.
        """
        
        # Iterate over the modules
        return iter(self._modules.values())
    
    
    @overload
    def __getitem__(self, idx: int) -> Module: 
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
    def __getitem__(self, idx: slice) -> list[Module]: 
        """
        Returns a list of modules at the specified slice.
        
        Parameters:
        - idx (slice): The slice of the modules to be returned.
        
        Returns:
        - list[Module]: A list of modules at the specified slice.
        """
        
        # Just to satisfy the overload
        ...
    
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, list[Module]]:
        """
        Returns the module at the specified index or a list of modules at the specified slice.
        
        Parameters:
        - idx (Union[int, slice]): The index or slice of the modules to be returned.
        
        Returns:
        - Union[Module, list[Module]]: The module at the specified index or a list of modules at the specified slice.
        """
        
        # Return the module at the specified index
        return list(self._modules.values())[idx]
    
    
    def __len__(self) -> int:
        """
        Returns the number of modules in the ModuleList.
        
        Returns:
        - int: The number of modules in the ModuleList.
        """
        
        # Return the length of the modules list
        return len(self._modules.values())
    
    
    ### Public Methods ###
    
    def train(self) -> None:
        """
        Sets the modules in the ModuleList to training mode.
        """
        
        # Set each module to training mode
        for module in self._modules.values():
            module.train()

            
    def eval(self) -> None:
        """
        Sets the modules in the ModuleList to evaluation mode.
        """
        
        # Set each module to evaluation mode
        for module in self._modules.values():
            module.eval()
            

    ### Protected Methods ###
    
    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass of the ModuleList.
        
        Parameters:
        - x (Tensor): The input tensor.
        
        Returns:
        - Tensor: The output tensor.
        """
        
        # Iterate over the modules
        for module in self._modules.values():
            # Perform the forward pass
            x = module.forward(x, *args, **kwargs)
            
        # Return the output tensor
        return x