from typing import Optional

from ...activations import ReLU
from ...core import Tensor, Module
from ...layers import Dense, Dropout


class MLP(Module):
    
    ### Magic methods ###
    
    def __init__(self, dropout: float = 0.1, name: Optional[str] = None) -> None:
        """
        Initialize the MLP layers of the transformer.
        
        Parameters:
        - dropout (float): The dropout rate.
        - name (Optional[str]): The name of the module.
        """
        
        # Initialize the superclass
        super().__init__(name)
        
        # Define the MLP layers
        # Define the dense layers
        # This will be lazily initialized in the forward pass, since we do not know the embedding size yet
        self.input_dense: Dense # (B, S, E) -> (B, S, 4 * E)
        self.output_dense: Dense # (B, S, 4 * E) -> (B, S, E)
        
        # Final dropout layer
        self.dropout: Dropout = Dropout(dropout) # (B, S, E) -> (B, S, E)
       
       
    ### Public methods ### 
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MLP layers.
        
        Parameters:
        - x (Tensor): The input tensor.
        
        Returns:
        - Tensor: The output embeddings.
        """
        
        # Check if the input shape is valid
        assert len(x.shape()) == 3, f"Invalid input shape. Input must be a 3D array. The shape must be (Batch size, sequence length, embedding size). Got shape: {x.shape()}"
        
        # Dimensions are:
        # - B: batch size
        # - S: sequence length
        # - E: embedding size (embedding dimension of the original data)
        
        # Store the input shape of the layer
        self.input_shape = x.shape()
        
        # Unpack the shape of the input data for better readability
        B, S, E = self.input_shape 
        
        # Check if the layer is initialized
        if not self.initialized:
            # Initialize the layer
            self.init_params(E)
            
        # Apply the forward pass through the MLP
        out = self.input_dense(x) # (B, S, E) -> (B, S, 4 * E)
        out = self.dropout(self.output_dense(out)) # (B, S, 4 * E) -> (B, S, E)
            
        # Return the output embeddings
        return out
        
        
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the module
        
        Returns:
        - tuple: The shape of the output of the module
        """
        
        # Call the parent class method to check if the layer is initialized
        super().output_shape()
        
        # Return the output shape
        return self.input_shape # (B, S, E)


    def init_params(self, E: int) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - E (int): The embedding size of the input data
        """
        
        # Initialize the dense layers
        self.input_dense = Dense(4 * E, activation=ReLU()) # (B, S, E) -> (B, S, 4 * E)
        self.output_dense = Dense(E) # (B, S, 4 * E) -> (B, S, E)
        
        # Call the parent class method to set the layer as initialized
        super().init_params()