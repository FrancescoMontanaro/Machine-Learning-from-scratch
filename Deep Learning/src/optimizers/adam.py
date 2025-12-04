import numpy as np
from typing import Optional

from ..core import Tensor
from .base import Optimizer
from ..core.utils.context_manager import no_grad
    

class Adam(Optimizer):
        
    ### Magic methods ###
        
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.00, parameters: Optional[list[Tensor]] = None) -> None:
        """
        Class constructor
        
        Parameters:
        - learning_rate (float): Learning rate for the optimizer
        - beta1 (float): Exponential decay rate for the first moment estimates
        - beta2 (float): Exponential decay rate for the second moment estimates
        - epsilon (float): Small value to prevent division by zero
        - weight_decay (float): Weight decay for the optimizer
        - parameters (Optional[list[Tensor]]): List of tensors containing the parameters of the optimizer
        """
        
        # Initialize the parent class
        super().__init__(parameters)
        
        # Store the parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        
    ### Public methods ###
    
    def update(self) -> None:
        """
        Method to update the parameters of the model
        """
        
        # Disable gradient computation
        with no_grad():
            # Iterate over the parameters and update them
            for param in self.parameters:
                # Get the name and id of the parameter
                param_id = id(param)
                
                # Check if the parameter has a gradient
                if param.grad is None:
                    # Skip parameters without gradient (they might not have been used in the forward pass)
                    continue
                
                # Check if state of the optimizer is initialized for the parameter
                if param_id not in self.state or len(self.state[param_id]) == 0:
                    self.state[param_id] = {
                        "m": np.zeros_like(param.grad),
                        "v": np.zeros_like(param.grad),
                        "t": 0
                    }
                    
                # Get the moments and time step from the state
                m = self.state[param_id]["m"]
                v = self.state[param_id]["v"]
                t = self.state[param_id]["t"]
                
                # Update the time step
                t += 1
                
                # Compute the first and second moment estimates
                m = self.beta1 * m + (1 - self.beta1) * param.grad
                v = self.beta2 * v + (1 - self.beta2) * (param.grad ** 2)
                
                # Compute the bias-corrected first and second moment estimates
                m_hat = m / (1 - self.beta1 ** t)
                v_hat = v / (1 - self.beta2 ** t)
                
                # Save updated moments and time step to the state
                self.state[param_id]["m"] = m
                self.state[param_id]["v"] = v
                self.state[param_id]["t"] = t
                
                # Update the parameters
                param.data -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * param.data)