from typing import Generic, TypeVar, Union, Optional, Iterator, overload, cast

from . import Tensor
from .module import Module
from .module_output import ModuleOutput

# Define a type variable T that is bound to the Module class
T = TypeVar('T', bound=Module)


class ModuleList(Module, Generic[T]):
    
    ### Magic Methods ###
    
    def __init__(self, modules: list[T], *args, **kwargs) -> None:
        """
        Class constructor.
        
        Parameters:
        - modules (list[T]): List of modules to be stored in the ModuleList.
        
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

    
    def __iter__(self) -> Iterator[T]:
        """
        Returns an iterator over the modules in the ModuleList.
        
        Returns:
        - Iterator[T]: An iterator over the modules in the ModuleList.
        """
        
        # Iterate over the modules (cast needed because _modules stores Module, not T)
        return cast(Iterator[T], iter(self._modules.values()))
    
    
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
        - Union[T, list[T]]: The module at the specified index or a list of modules at the specified slice.
        """
        
        # Get the module(s) at the specified index or slice
        out = list(self._modules.values())[idx]
        out = cast(Union[T, list[T]], out)

        # Return the output
        return out
    
    
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
    
    def _forward(self, **kwargs) -> Union[Tensor, 'ModuleOutput']:
        """
        Forward pass through all modules in sequence.
        
        Parameters:
        - **kwargs: Named input tensors and additional parameters
        
        Returns:
        - Union[Tensor, ModuleOutput]: The output of the last module after processing through all modules in sequence.
          Returns the raw result (Tensor or ModuleOutput) - Module.forward() handles wrapping.
        
        Note:
        - All arguments must be passed as keyword arguments
        - The first module receives all kwargs
        - Subsequent modules receive the primary output (.output) of the previous module as 'x' plus any non-tensor kwargs
        """
        
        # Separate tensor inputs from other params (non-tensor params are passed to all modules)
        other_params = {k: v for k, v in kwargs.items() if not isinstance(v, Tensor)}
        
        # Initialize the current output
        current_output: Optional[ModuleOutput] = None
        
        # Iterate through the modules
        for i, module in enumerate(self._modules.values()):
            if i == 0:
                # First module gets all named inputs
                current_output = module.forward(**kwargs)
            else:
                # Subsequent modules get the primary tensor from the previous output as 'x'
                assert current_output is not None
                current_output = module.forward(x=current_output.output, **other_params)
            
            # Validate output type
            if not isinstance(current_output, ModuleOutput):
                raise ValueError(f"Module {i} must return a ModuleOutput, got {type(current_output)}")
                
        # Return the final output after processing through all modules
        if current_output is None:
            raise ValueError("No modules in the list to process.")
        
        # Return the raw output (Tensor .output) so that the parent Module.forward() can wrap it
        return current_output