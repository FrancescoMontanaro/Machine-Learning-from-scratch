from typing import Any

from ..core import Module


class Callback:
    
    ### Magic methods ###
    
    def __call__(self, module: Module) -> Any:
        """
        This method is called when the callback is called.
        
        Parameters:
        - module (Module): The module that is being trained.
        
        Returns:
        - Any: the output of the callback.
        """
        
        pass