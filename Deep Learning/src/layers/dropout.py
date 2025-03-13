import numpy as np
from typing import Optional

from ..core import Tensor, Module


class Dropout(Module):
    
    ### Magic methods ###
    
    def __init__(self, rate: float, name: Optional[str] = None) -> None:
        """
        Initialize the dropout layer.
        
        Parameters:
        - rate (float): The dropout rate.
        - name (str): The name of the layer.
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Initialize the dropout rate
        self.rate = rate
        self.mask = None
    
    
    ### Public methods ###
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the dropout layer.
        
        Parameters:
        - x (Tensor): The input tensor.
        
        Returns:
        - Tensor: The output tensor.
        """
        
        # Store the input shape
        self.input_shape = x.shape()
        
        # Initialize the layer params
        if not self.initialized:
            self.init_params()
        
        # Generate a random mask
        self.mask = Tensor(
            data = np.random.rand(*x.shape()) > self.rate, 
            requires_grad = False, 
            is_parameter = False
        )
        
        if self.training:
            # Scale the output during training
            return x * self.mask / (1 - self.rate)
        else:
            # Return the output during inference
            return x
    
    
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tuple: The shape of the output of the layer
        """
        
        # Return the output shape of the layer
        return self.input_shape