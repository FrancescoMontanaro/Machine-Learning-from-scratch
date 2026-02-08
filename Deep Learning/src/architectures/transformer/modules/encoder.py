import numpy as np

from .block import Block
from ..config import TransformerConfig
from ....core import Tensor, Module, ModuleList
from ....layers import Dense, Embedding, LayerNormalization, PositionalEncoding


class Encoder(Module):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        config: TransformerConfig,
        *args, **kwargs
    ) -> None:
        """
        Initialize the transformer's encoder.
        
        Parameters:
        - config (TransformerConfig): The configuration for the encoder transformer model.
        """
        
        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        # Save configuration parameters
        self.return_sequence = config.return_sequence
        
        # Define the input projection layer
        if config.input_type == "discrete":
            self.input_proj = Embedding(config.input_dim, config.embed_dim) # (B, S) -> (B, S, E)
        else:
            self.input_proj = Dense(config.embed_dim) # (B, S, F) -> (B, S, E)

        # Define the positional embedding layer
        if config.positional_encoding_type == "learned":
            self.positional_encoding = Embedding(config.max_sequence_length, config.embed_dim) # (B, S) -> (B, S, E)
        else:
            self.positional_encoding = PositionalEncoding(config.max_sequence_length) # (B, S) -> (B, S, E)

        # Instantiate the encoder blocks
        self.encoder_blocks: ModuleList[Block] = ModuleList([ # (B, S, E) -> (B, S, E)
            Block(config=config.block_config)
            for _ in range(config.num_blocks)
        ])
        
        # Layer normalization
        self.layer_norm: LayerNormalization = LayerNormalization() # (B, S, E) -> (B, S, E)
        
        
    ### Protected methods ###
    
    def _forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the transformer's encoder.
        
        Parameters:
        - x (Tensor): The input x.
        
        Returns:
        - Tensor: The encoded representations.
        """
        
        # Dimensions are:
        # - B: batch size
        # - S: sequence length
        # - E: embedding size (the dimension of the embedding space)
        
        # Unpack the shape of the input data for better readability
        _, S, *_ = x.shape # (B, S)
            
        # Project the input data to the embedding space
        embeddings = self.input_proj(x).output # (B, S) -> (B, S, E)
        
        # Add positional encoding
        if isinstance(self.positional_encoding, PositionalEncoding):
            # Use sinusoidal positional encoding (adds directly to embeddings)
            embeddings = self.positional_encoding(embeddings).output # (B, S, E) -> (B, S, E)
        else:
            # Use trainable positional embeddings
            positions = Tensor(np.arange(S)) # Create position indices (S,)
            pos_embeddings = self.positional_encoding(positions).output # (S,) -> (S, E)
            
            # Add positional embeddings to input embeddings
            embeddings = embeddings + pos_embeddings # (B, S, E) + (S, E) -> (B, S, E)
        
        # Apply the encoder blocks
        for block in self.encoder_blocks:
            embeddings = block(embeddings).output # (B, S, E) -> (B, S, E)
            
        # Apply layer normalization and return the encoded representations
        out = self.layer_norm(embeddings).output # (B, S, E) -> (B, S, E)
        
        # If the model is set to return the full sequence, return the output
        if self.return_sequence:
            return out
        
        # If the model is not set to return the full sequence, return only the last output
        return out[:, -1:, ...] # (B, S, E) -> (B, 1, E)