import numpy as np
from typing import Any

from .base import Optimizer


class SGD(Optimizer):
    
    ### Magic methods ###
    
    def __init__(self, learning_rate: float, momentum: float = 0.0) -> None:
        """
        Class constructor for the Stochastic Gradient Descent (SGD) optimizer.
        Momentum accelerates updates along directions where the gradients are consistent 
        and reduces oscillations as the optimizer approaches the minimum of the loss function. 
        This allows the optimizer to converge more quickly while also reducing the risk of 
        oscillating around the minimum.
        
        Parameters:
        - learning_rate (float): Learning rate for the optimizer
        - momentum (float): Momentum for the optimizer
        """
        
        # Initialize the parent class
        super().__init__()
        
        # Store the learning rate and momentum
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    
    ### Public methods ###
    
    def update(self, layer: Any, param_name: str, params: np.ndarray, grad_params: np.ndarray) -> np.ndarray:
        """
        Method to update the parameters of the model.
        
        Parameters:
        - layer (Any): Instance of the Layer being optimized
        - param_name (str): Name of the parameters to be updated
        - params (np.ndarray): Parameters of the model
        - grad_params (np.ndarray): Gradient of the parameters
        
        Returns:
        - np.ndarray: Updated parameters
        """
        
        # Getting the layer id
        layer_id = id(layer)
        
        # Initialize the layer registry if missing
        if layer_id not in self.params_registry:
            self.params_registry[layer_id] = {}
        
        # Initialize velocity for the layer if missing
        if param_name not in self.params_registry[layer_id]:
            self.params_registry[layer_id][param_name] = {}

        # Initialize specific parameter if missing
        if "velocity" not in self.params_registry[layer_id][param_name]:
            self.params_registry[layer_id][param_name]["velocity"] = np.zeros_like(grad_params)
            
        # Get the velocity from the registry
        velocity = self.params_registry[layer_id][param_name]["velocity"]
            
        # Update the velocity
        velocity = self.momentum * velocity - self.learning_rate * grad_params
        
        # Save updated velocity to the registry
        self.params_registry[layer_id][param_name]["velocity"] = velocity
        
        # Update the parameters
        return params + velocity