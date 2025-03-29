from typing import Optional

from .mlp import MLP
from ...core import Tensor, Module
from ...layers import LayerNormalization
from .attention_mechanism import MultiHeadAttention


class DecoderBlock(Module):
    
    ### Magic methods ###
    
    def __init__(self, n_heads: int, dropout: float = 0.1, name: Optional[str] = None) -> None:
        """
        Initialize the transformer's decoder block.
        
        Parameters:
        - n_heads (int): The number of attention heads.
        - dropout (float): The dropout rate.
        - name (Optional[str]): The name of the module.
        """
        
        # Initialize the superclass
        super().__init__(name)
        
        # Store the number of heads and dropout rate
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Define the modules of the transformer's decoder block
        # Some of them will be lazily initialized in the forward pass, since we do not know the embedding size yet
        self.layer_norm_1 = LayerNormalization() # (B, S, E) -> (B, S, E)
        self.attention_heads: MultiHeadAttention # (B, S, E) -> (B, S, E)
        self.mlp: MLP = MLP(dropout) # (B, S, E) -> (B, S, E)
        self.layer_norm_2 = LayerNormalization() # (B, S, E) -> (B, S, E)
      
    
    ### Public methods ###  
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the transformer block.
        
        Parameters:
        - x (Tensor): The input x.
        
        Returns:
        - Tensor: The output embeddings.
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
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
        
        # Apply the multi-attention mechanism with skip connections
        out = x + self.attention_heads(self.layer_norm_1(x)) # (B, S, E) + (B, S, E) -> (B, S, E)
        
        # Apply the MLP modlue with skip connections
        out = out + self.mlp(self.layer_norm_2(out)) # (B, S, E) + (B, S, E) -> (B, S, E)
        
        # Return the output embeddings
        return out # (B, S, E)
    
    
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
        
        # A good head size is the embedding size divided by the number of heads
        head_size = E // self.n_heads
        
        # Initialize the multi-head attention mechanism
        self.attention_heads = MultiHeadAttention( # (B, S, E) -> (B, S, E)
            n_heads = self.n_heads, 
            head_size = head_size, 
            dropout = self.dropout
        )
        
        # Call the parent class method to set the layer as initialized
        super().init_params()