from typing import Any

from ..neural_networks import FeedForward


class Callback:
    
    ### Magic methods ###
    
    def __call__(self, model_instance: FeedForward) -> Any:
        """
        This method is called when the callback is called.
        
        Parameters:
        - model_instance: FeedForward: The model instance that is being trained.
        
        Returns:
        - Any: the output of the callback.
        """
        
        pass