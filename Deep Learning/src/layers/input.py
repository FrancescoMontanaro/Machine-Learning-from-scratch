from typing import Optional

from ..core import Tensor, Module


class Input(Module):
    
    ### Magic methods ###
    
    def __init__(self, shape: tuple, name: Optional[str] = None) -> None:
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
    
    
    ### Public methods ###
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Function to compute the forward pass of the Input layer.
        
        Parameters:
        - x (Tensor): input data
        
        Returns:
        - Tensor: output data
        """
        
        # Save the input shape
        self.input_shape = x.shape()
        
        # Checking if the layer is initialized
        if not self.initialized:
            # Initialize the filters
            self.init_params()
        
        # Return the input data as it is
        return x
    
    
    def output_shape(self) -> tuple:
        """
        Method to get the output shape of the layer.
        
        Returns:
        - tuple: output shape of the layer
        """
        
        # Call the parent class method to check if the layer is initialized
        super().output_shape()
        
        # Return the input shape
        return self.input_shape