from typing import Optional

from ..core import Tensor


class Optimizer:
    
    ### Magic methods ###
    
    def __init__(self, parameters: Optional[list[Tensor]] = None) -> None:
        """
        Class constructor
        
        Parameters:
        - parameters (Optional[list[Tensor]]): List of tensors containing the parameters of the optimizer
        """
        
        self.parameters: list[Tensor] = [] # Dictionary to store the parameters of the optimizer
        self.state = {id(p): {} for p in self.parameters} # Dictionary to store the state parameters of the optimizer (e.g., momentum, velocity)
        
        # Set the parameters of the optimizer if provided
        if parameters is not None:
            self.set_parameters(parameters)
        
    
    ### Public methods ###

    def update(self) -> None:
        """
        Method to update the parameters of the model
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method 'update' is not implemented.")
    
    
    def zero_grad(self) -> None:
        """
        Method to zero the gradients of the parameters
        """
        
        # Iterate over the parameters and zero the gradients
        for parameter in self.parameters:
            # Zero the gradients of the parameter
            parameter.zero_grad()
    
    
    def set_parameters(self, parameters: list[Tensor]) -> None:
        """
        Method to set the parameters of the model
        
        Parameters:
        - parameters (list[dict[str, Tensor]]): List of dictionaries containing the parameters of the model
        """
        
        # Set the parameters of the model
        self.parameters = parameters
        self.state = {id(p): {} for p in self.parameters}


    def init_parameters(self) -> None:
        """
        Method to initialize the parameters of the optimizer
        """
        
        # Reset the parameters and state of the optimizer
        self.parameters = []
        self.state = {}
        