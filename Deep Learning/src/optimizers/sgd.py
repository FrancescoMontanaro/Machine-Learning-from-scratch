import numpy as np
from typing import Optional

from ..core import Tensor
from .base import Optimizer
from ..core.context_manager import no_grad


class SGD(Optimizer):
    
    ### Magic methods ###
    
    def __init__(self, learning_rate: float, momentum: float = 0.0, parameters: Optional[list[Tensor]] = None) -> None:
        """
        Class constructor for the Stochastic Gradient Descent (SGD) optimizer.
        Momentum accelerates updates along directions where the gradients are consistent 
        and reduces oscillations as the optimizer approaches the minimum of the loss function. 
        This allows the optimizer to converge more quickly while also reducing the risk of 
        oscillating around the minimum.
        
        Parameters:
        - learning_rate (float): Learning rate for the optimizer
        - momentum (float): Momentum for the optimizer
        - parameters (Optional[list[Tensor]]): List of tensors containing the parameters of the optimizer
        """
        
        # Initialize the parent class
        super().__init__(parameters)
        
        # Store the learning rate and momentum
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    
    ### Public methods ###
    
    def update(self) -> None:
        """
        Method to update the parameters of the model.
        """
        
        # Disable gradient computation
        with no_grad():
            # Iterate over the parameters and update them
            for param in self.parameters:
                # Get the name and id of the parameter
                param_id = id(param)
                
                # Check if the parameter has a gradient
                if param.grad is None:
                    # Raise an error if the gradient is missing
                    raise ValueError(f"Impossible to update the parameters. Missing gradient for the parameter {param_id}")
                
                # Check if state of the optimizer is initialized for the parameter
                if param_id not in self.state or len(self.state[param_id]) == 0:
                    self.state[param_id] = {
                        "velocity": np.zeros_like(param.grad)
                    }
                
                # Axtract the velocity from the registry
                velocity = self.state[param_id]["velocity"]
                    
                # Update the velocity
                velocity = self.momentum * velocity - self.learning_rate * param.grad
                
                # Save updated velocity to the registry
                self.state[param_id]["velocity"] = velocity
                
                # Update the parameters
                param.data += velocity