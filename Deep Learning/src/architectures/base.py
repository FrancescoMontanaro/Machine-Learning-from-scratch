import gc
import numpy as np
from typing import Callable

from ..core import Module, Tensor


class Architecture(Module):
    
    ### Magic methods ###
    
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the architecture.
        """
        
        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        # Initialize the history of the model
        self.history = {}
        
        
    ### Public methods ###
    
    def init_history(self, metrics: list[Callable] = []) -> None:
        """
        Method to initialize the history of the model
        
        Parameters:
        - metrics (list[Callable]): List of metrics to evaluate the model
        """
        
        # Initialize the history of the model
        self.history = {
            "loss": Tensor(np.array([])),
            **{f"{metric.__name__}": Tensor(np.array([])) for metric in metrics},
            "val_loss": Tensor(np.array([])),
            **{f"val_{metric.__name__}": Tensor(np.array([])) for metric in metrics}
        }
    
    
    def fit(self, *args, **kwargs) -> dict[str, Tensor]:
        """
        Method to fit the model.
        
        Returns:
        - dict[str, Tensor]: Dictionary containing the history of the model
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method 'fit' is not implemented. Please implement it in the child class.")
    
    
    def clear_cache(self) -> None:
        """
        Method to clear the cache of the model.
        """
        
        # Clear the cache of the model by calling the garbage collector
        gc.collect()
        
        
    def count_tensors_in_memory(self) -> int:
        """
        Method to get the tensors in memory.
        
        Returns:
        - int: Number of tensors in memory
        """
        
        # Count the number of tensors in memory using the garbage collector
        return len([t for t in gc.get_objects() if isinstance(t, Tensor)])