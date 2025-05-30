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
    
    def _forward(self, *args, **kwargs) -> Tensor:
        """
        Forward pass through all modules in sequence
        """
        
        # Determine input method
        use_named_inputs = bool(kwargs and any(isinstance(v, Tensor) for v in kwargs.values()))
        
        # If using named inputs, ensure all inputs are tensors
        if use_named_inputs:
            # Named inputs - separate tensor inputs from other params
            other_params = {k: v for k, v in kwargs.items() if not isinstance(v, Tensor)}
            
            # Create a list of tensor inputs
            current_output = None
            
            # Iterate through the modules
            for i, module in enumerate(self._modules.values()):
                if i == 0:
                    # First module gets all named inputs
                    current_output = module.forward(**kwargs)
                else:
                    # Subsequent modules get single tensor output + other params
                    # Assuming the output key name doesn't matter for subsequent modules
                    current_output = module.forward(current_output, **other_params)
                
                # Ensure the output is a Tensor
                if not isinstance(current_output, Tensor):
                    # If the output is not a Tensor, raise an error
                    raise ValueError(f"Module {i} must return a Tensor, got {type(current_output)}")
        
        else:
            # Positional inputs - same as before
            other_params = [arg for arg in args if not isinstance(arg, Tensor)]
            
            # Create a list of tensor inputs
            current_output = None
            
            # Iterate through the modules
            for i, module in enumerate(self._modules.values()):
                if i == 0:
                    # First module gets all original inputs
                    current_output = module.forward(*args)
                else:
                    # Subsequent modules get single tensor output + other params
                    module_args = [current_output] + other_params
                    current_output = module.forward(*module_args)
                
                # Ensure the output is a Tensor
                if not isinstance(current_output, Tensor):
                    raise ValueError(f"Module {i} must return a Tensor, got {type(current_output)}")
                
        # Return the final output after processing through all modules
        if not isinstance(current_output, Tensor):
            raise ValueError("Final output must be a Tensor.")
        
        # Return the final output tensor
        return current_output