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
    
    
class Adam(Optimizer):
        
    ### Magic methods ###
        
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.00) -> None:
        """
        Class constructor
        
        Parameters:
        - learning_rate (float): Learning rate for the optimizer
        - beta1 (float): Exponential decay rate for the first moment estimates
        - beta2 (float): Exponential decay rate for the second moment estimates
        - epsilon (float): Small value to prevent division by zero
        - weight_decay (float): Weight decay for the optimizer
        """
        
        # Initialize the parent class
        super().__init__()
        
        # Store the parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        
    ### Public methods ###
    
    def update(self, layer: Any, param_name: str, params: np.ndarray, grad_params: np.ndarray) -> np.ndarray:
        """
        Method to update the parameters of the model
        
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
            
        # Initialize the moments
        if "moments" not in self.params_registry[layer_id][param_name]:
            self.params_registry[layer_id][param_name]["moments"] = {
                "m": np.zeros_like(grad_params),
                "v": np.zeros_like(grad_params)
            }
            
        # Initialize the time step
        if "t" not in self.params_registry[layer_id][param_name]:
            self.params_registry[layer_id][param_name]["t"] = 0
        
        # Get the moments and time step from the registry
        m = self.params_registry[layer_id][param_name]["moments"]["m"]
        v = self.params_registry[layer_id][param_name]["moments"]["v"]
        t = self.params_registry[layer_id][param_name]["t"]
        
        # Update the time step
        t += 1
        
        # Compute the first and second moment estimates
        m = self.beta1 * m + (1 - self.beta1) * grad_params
        v = self.beta2 * v + (1 - self.beta2) * (grad_params ** 2)
        
        # Compute the bias-corrected first and second moment estimates
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)
        
        # Save updated moments and time step to the registry
        self.params_registry[layer_id][param_name]["moments"]["m"] = m
        self.params_registry[layer_id][param_name]["moments"]["v"] = v
        self.params_registry[layer_id][param_name]["t"] = t
        
        # Update the parameters
        return params - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * params)