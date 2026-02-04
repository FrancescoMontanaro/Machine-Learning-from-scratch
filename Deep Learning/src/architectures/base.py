import gc
from typing import Callable, Dict, TYPE_CHECKING

from ..core import Module, Tensor
from ..core.utils.progress_printer import ProgressPrinter


class Architecture(Module):
    
    ### Magic methods ###
    
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the architecture.
        """
        
        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        # Initialize the history of the model
        self.history: Dict[str, list[float]] = {}
        
        # Initialize the progress printer
        self._progress_printer = ProgressPrinter()
        
    ### Properties ###    
    
    @property
    def tensors_in_memory(self) -> int:
        """
        Method to get the tensors in memory.
        
        Returns:
        - int: Number of tensors in memory
        """
        
        # Count the number of tensors in memory using the garbage collector
        return Tensor.count_live()
        
        
    ### Public methods ###
    
    def init_history(self, metrics: list[Callable] = []) -> None:
        """
        Method to initialize the history of the model
        
        Parameters:
        - metrics (list[Callable]): List of metrics to evaluate the model
        """
        
        # Initialize the history of the model
        self.history = {
            "loss": [],
            **{f"{metric.__name__}": [] for metric in metrics},
            "val_loss": [],
            **{f"val_{metric.__name__}": [] for metric in metrics}
        }
    
    
    def fit(self, *args, **kwargs) -> Dict[str, list[float]]:
        """
        Method to fit the model.
        
        Returns:
        - Dict[str, list[float]]: Dictionary containing the history of the model
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method 'fit' is not implemented. Please implement it in the child class.")
    
    
    def clear_cache(self) -> None:
        """
        Method to clear the cache of the model.
        """
        
        # Clear the cache of the model by calling the garbage collector
        gc.collect()



class SingleOutputArchitecture(Architecture):
    """
    Class representing an architecture that outputs a single tensor.
    """
    
    if TYPE_CHECKING:
        def __call__(self, *args, **kwargs) -> Tensor: ...
        def _forward(self, *args, **kwargs) -> Tensor: ...
        def forward(self, *args, **kwargs) -> Tensor: ...


class MultiOutputArchitecture(Architecture):
    """
    Class representing an architecture that outputs multiple tensors.
    """
    
    if TYPE_CHECKING:
        def __call__(self, *args, **kwargs) -> tuple[Tensor, ...]: ...
        def _forward(self, *args, **kwargs) -> tuple[Tensor, ...]: ...
        def forward(self, *args, **kwargs) -> tuple[Tensor, ...]: ...