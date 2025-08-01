import numpy as np
from typing import Literal, List

from .base import Callback
from ..core import Module


class EarlyStopping(Callback):
    
    ### Magic methods ###
    
    def __init__(self, monitor: str = "val_loss", patience: int = 5, min_delta: float = 0.0, mode: Literal["min", "max"] = "min") -> None:
        """
        Construct a new EarlyStopping callback.
        
        Parameters:
        - monitor (str): The target to monitor. Default is "val_loss".
        - patience (int): Number of epochs to wait before stopping the training.
        - min_delta (float): Minimum change in the monitored value to qualify as an improvement.
        """
        
        # Store the parameters
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        # Initialize the best value and counter
        self.best_value = np.inf if mode == "min" else -np.inf
        self.counter = 0
        
    
    def __call__(self, module: Module) -> bool:
        """
        Method to call the callback. This method will check if the training should be stopped by
        setting the stop_training attribute of the model instance.
        
        Parameters:
        - module (Module): The model instance that is being trained.
        
        Returns:
        - bool: True if the training should be stopped, False otherwise.
        
        Raises:
        - ValueError: If the target value is not found in the history.
        """
        
        # Get the history of the module
        history = getattr(module, "history")
        
        # Check if the history is None
        if history is None:
            # Raise an error
            raise ValueError("History not found in the module.")
        
        # Get the target value
        target: List[float] = history.get(self.monitor)
        
        # Check if the target value is None
        if target is None:
            raise ValueError(f"Target value '{self.monitor}' not found in the history.")
        
        # Get the current value
        value = target[-1]
        
        # Check if the value is better
        if self.mode == "min":
            # Check if the value is less than the best value minus the minimum delta
            is_better = value < self.best_value - self.min_delta
        else:
            # Check if the value is greater than the best value plus the minimum delta
            is_better = value > self.best_value + self.min_delta
            
        # Update the best value
        if is_better:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            
        # Check if the training should be stopped
        if self.counter >= self.patience:
            # Extract the epoch number
            epoch = len(target)
            
            # Print a message
            print(f"Early stopping: stopping training after {epoch} epochs.")
            
            # Set the stop_training attribute
            setattr(module, "stop_training", True)
            
            # The training should be stopped
            return True
        
        # The training should not be stopped
        return False