import numpy as np
from typing import Optional

from .base import Layer


class Dropout(Layer):
    
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
        
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Method to set the input shape of the layer.
        
        Parameters:
        - x (np.ndarray): Input data.
        
        Returns:
        - np.ndarray: Output of the layer after the forward pass.
        """
        
        # Store the input dimension
        self.input_shape = x.shape # (Batch size, number of features)
        
        # Initialize the layer params
        if not self.initialized:
            self.init_params()
        
        # Forward pass
        return self.forward(x)
    
    
    ### Public methods ###
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the dropout layer.
        
        Parameters:
        - x (np.ndarray): The input tensor.
        
        Returns:
        - np.ndarray: The output tensor.
        """
        
        # Generate the mask
        self.mask = np.random.rand(*x.shape) > self.rate
        
        if self.training:
            # Scale the output during training
            return x * self.mask / (1 - self.rate)
        else:
            # Return the output during inference
            return x
    
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of the dropout layer (layer i)
        
        Parameters:
        - grad_output (np.ndarray): The gradient of the loss with respect to the output of the layer: dL/dO_i
        
        Returns:
        - np.ndarray: The gradient of the loss with respect to the input of the layer: dL/dX_i ≡ dL/dO_{i-1}
        """
        
        # Scale the gradient by the mask
        return grad_output * self.mask / (1 - self.rate) # dL/dX_i ≡ dL/dO_{i-1}, to pass to the previous layer
    
    
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tuple: The shape of the output of the layer
        """
        
        return self.input_shape