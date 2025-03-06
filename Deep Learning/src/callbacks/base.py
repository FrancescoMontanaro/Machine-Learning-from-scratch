from typing import Any

from ..model import Model


class Callback:
    
    ### Magic methods ###
    
    def __call__(self, model_instance: Model) -> Any:
        """
        This method is called when the callback is called.
        
        Parameters:
        - model_instance: Model: The model instance that is being trained.
        
        Returns:
        - Any: the output of the callback.
        """
        
        pass