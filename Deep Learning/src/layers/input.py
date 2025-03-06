import numpy as np
from typing import Literal, Optional

from .base import Layer
from ..optimizers import Optimizer
from ..activations import Activation


class Input(Layer):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        shape: tuple, 
        name: Optional[str] = None
    ) -> None:
        """
        Class constructor for Input layer.
        
        Parameters:
        - shape (tuple): shape of the input data
        - name (str): name of the layer
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Initializing the input shape
        self.input_shape = shape
    
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Function to compute the forward pass of the Input layer.
        It initializes the filters if not initialized and computes the forward pass.
        
        Parameters:
        - x (np.ndarray): input data.
        
        Returns:
        - np.ndarray: output data.
        """
        
        # Save the input shape
        self.input_shape = x.shape
        
        # Checking if the layer is initialized
        if not self.initialized:
            # Initialize the filters
            self.init_params()
            
        # Compute the forward pass
        return self.forward(x)
    
    
    ### Public methods ###
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Function to compute the forward pass of the Input layer.
        
        Parameters:
        - x (np.ndarray): input data
        
        Returns:
        - np.ndarray: output data
        """
        
        # Return the input data as it is
        return x
    
    
    def backward(self, loss_gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass of the Input layer
        
        Parameters:
        - loss_gradient (np.ndarray): Gradient of the loss with respect to the output of the layer: dL/dO_i
        
        Returns:
        - np.ndarray: Gradient of the loss with respect to the input of the layer: dL/dX_i ≡ dL/dO_{i-1}
        """
        
        # Return the loss gradient as it is
        return loss_gradient
    
    
    def output_shape(self) -> tuple:
        """
        Method to get the output shape of the layer.
        
        Returns:
        - tuple: output shape of the layer
        """
        
        return self.input_shape