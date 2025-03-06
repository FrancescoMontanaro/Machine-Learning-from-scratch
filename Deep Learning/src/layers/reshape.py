import numpy as np
from typing import Optional

from .base import Layer


class Reshape(Layer):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        shape: tuple, 
        name: Optional[str] = None
    ) -> None:
        """
        Class constructor for Reshape layer.
        
        Parameters:
        - shape (tuple): target shape of the input data
        - name (str): name of the layer
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Initializing the input shape
        self.input_shape = None
        
        # Save the target shape
        self.target_shape = shape
    
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Function to compute the forward pass of the Reshape layer.
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
        Function to compute the forward pass of the Reshape layer.
        
        Parameters:
        - x (np.ndarray): input data
        
        Returns:
        - np.ndarray: output data
        """
        
        # Extract the batch size
        batch = x.shape[0]
        
        # Reshape the input data to the target shape
        # The batch size is kept the same
        return x.reshape((batch, *self.target_shape))
    
    
    def backward(self, loss_gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass of the Reshape layer
        
        Parameters:
        - loss_gradient (np.ndarray): Gradient of the loss with respect to the output of the layer: dL/dO_i
        
        Returns:
        - np.ndarray: Gradient of the loss with respect to the input of the layer: dL/dX_i ≡ dL/dO_{i-1}
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
        # Check if the input shape is set
        assert self.input_shape is not None, "Input shape is not set. Please call the layer with some input data to set the input shape."
        
        # Reshape the loss gradient to the input shape and return it
        return loss_gradient.reshape(self.input_shape)
    
    
    def output_shape(self) -> tuple:
        """
        Method to get the output shape of the layer.
        
        Returns:
        - tuple: output shape of the layer
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
        # Assert that the input shape is set
        assert self.input_shape is not None, "Input shape is not set. Please call the layer with some input data to set the input shape."
        
        # Extract the batch size
        batch = self.input_shape[0]
        
        # Return the output shape
        # The batch size is kept the same
        return (batch, *self.target_shape)