import numpy as np
from typing import Optional

from ..core import Tensor, Module


class Embedding(Module):
    
    ### Magic methods ###
    
    def __init__(self, input_dim: int, output_dim: int, name: Optional[str] = None) -> None:
        """
        Class constructor
        
        Parameters:
        - input_dim (int): Number of features in the input data (e.g. vocabulary size)
        - output_dim (int): Embedding dimension
        - name (str): Name of the layer
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Store the activation function
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize the Embedding matrix
        self.embedding: Tensor
    
    
    ### Public methods ###

    def forward(self, x: Tensor) -> Tensor:
        """
        Foward pass of the layer
        
        Parameters:
        - x (Tensor): Features of the dataset
        
        Returns:
        - Tensor: Output of the layer
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        - AssertionError: If the Embedding matrix is not initialized
        """
        
        # Check if the input shape is valid
        assert len(x.shape()) <= 2, f"Invalid input shape. Input must be at maximum a 2D array. Got shape: {x.shape()}"
        
        # Check if the input is 1D or 2D
        self.was_1d = len(x.shape()) == 1
        
        # If the input is 1D, reshape it to 2D
        if self.was_1d:
            # reshape the input to 2D
            x = x.reshape((1, -1)) 
        
        # Save the input shape
        self.input_shape = x.shape()
        
        # Check if the layer is initialized
        if not self.initialized:
            # Initialize the layer
            self.init_params()
        
        # Assert that the Embedding matrix is initialized
        assert isinstance(self.embedding, Tensor), "Embeddings are not initialized. Please call the layer with some input data to initialize the embedding matrix."
    
        # Return the embeddings
        out = self.embedding[x.data.astype(int)]
        
        # Squeeze the output if the input was 1D and return the it
        return out.squeeze(0) if self.was_1d else out

    
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tuple: The shape of the output of the layer
        """
        
        # Call the parent class method to check if the layer is initialized
        super().output_shape()
        
        # Unpack the input shape for better readability
        batch_size, _ = self.input_shape
        
        # The output shape
        return (batch_size, self.output_dim) if not self.was_1d else (self.output_dim,)
    
        
    def init_params(self) -> None:
        """
        Method to initialize the parameters of the layer
        """
        
        # Initialize the embedding matrix
        self.embedding = Tensor(
            data = np.random.uniform(-1, 1, (self.input_dim, self.output_dim)),
            requires_grad = True,
            is_parameter = True
        )
        
        # Call the parent class method to set the layer as initialized
        super().init_params()