import numpy as np
from typing import Any


class Optimizer:
    
    ### Magic methods ###
    
    def __init__(self) -> None:
        """
        Class constructor
        """
        
        # Create a dictionary to store the parameters of the model
        self.params_registry = {}
        
    
    ### Public methods ###

    def update(self, layer: Any, param_name: str, params: np.ndarray, grad_params: np.ndarray) -> np.ndarray:
        """
        Abstract method to update the parameters of the model
        
        Parameters:
        - layer (Any): Instance of the Layer being optimized
        - param_name (str): Name of the parameters to be updated
        - params (np.ndarray): Parameters to be updated
        - grad_params (np.ndarray): Gradient of the parameters with respect to the loss
        
        Returns:
        - np.ndarray: Updated parameters
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method 'update' is not implemented.")


    def init_params_registry(self) -> None:
        """
        Method to initialize the parameters of the optimizer
        """
        
        # Reset the parameters registry
        self.params_registry = {}