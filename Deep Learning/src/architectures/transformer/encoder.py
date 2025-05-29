import numpy as np
from typing import Literal

from .block import Block
from ...core import Tensor, Module, ModuleList
from ...layers import Dense, Embedding, LayerNormalization


class Encoder(Module):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        input_dim: int, 
        n_embed: int, 
        n_attention_heads: int, 
        sequence_length: int, 
        n_encoder_blocks: int = 4, 
        dropout: float = 0.1,
        input_type: Literal["discrete", "continuous"] = "discrete",
        causal_attention: bool = False,
        return_sequence: bool = False,
        *args, **kwargs) -> None:
        """
        Initialize the transformer's encoder.
        
        Parameters:
        - input_dim (int): The input dimension. It is the vocabulary size for text data or the number of features for other types of data.
        - n_embed (int): The embedding size.
        - n_attention_heads (int): The number of attention heads.
        - sequence_length (int): The sequence length of the input data.
        - n_encoder_blocks (int): The number of transformer blocks.
        - dropout (float): The dropout rate.
        - input_type (Literal["discrete", "continuous"]): The type of input data. It can be "discrete" for text data or "continuous" for other types of data.
        - causal_attention (bool): Whether to use causal attention (default: False).
        - return_sequence (bool): Whether to return the sequence or just the last output. Default is False.
        """
        
        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        # Store the parameters
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.n_attention_heads = n_attention_heads
        self.sequence_length = sequence_length
        self.n_encoder_blocks = n_encoder_blocks
        self.dropout = dropout
        self.causal_attention = causal_attention
        self.return_sequence = return_sequence
        
        # Define the input projection layer
        if input_type == "discrete":
            self.input_proj = Embedding(input_dim, n_embed) # (B, S) -> (B, S, E)
        else:
            self.input_proj = Dense(n_embed) # (B, S, F) -> (B, S, E)
        
        # Define the positional embedding layer
        self.positional_embedding: Embedding = Embedding(sequence_length, n_embed) # (S) -> (S, E)
        
        # Instantiate the encoder blocks
        self.encoder_blocks: ModuleList[Block] = ModuleList([ # (B, S, E) -> (B, S, E)
            Block(
                n_heads = n_attention_heads, 
                dropout = dropout,
                causal_attention = causal_attention, # Causal attention is usually not used in the encoder
            ) 
            for _ in range(n_encoder_blocks)
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
        _, S, *_ = x.shape() # (B, S)
            
        # Project the input data to the embedding space
        embeddings = self.input_proj(x) # (B, S) -> (B, S, E)
        
        # Add positional encoding
        positions = self.positional_embedding(Tensor(np.arange(S))) # (S) -> (1, S) -> (1, S, E)
        
        # Embed the input data and add the positional encoding
        embeddings = embeddings + positions # (B, S, E) + (S, E) -> (B, S, E)
        
        # Apply the encoder blocks
        for block in self.encoder_blocks:
            embeddings = block(embeddings) # (B, S, E) -> (B, S, E)
            
        # Apply layer normalization and return the encoded representations
        out = self.layer_norm(embeddings) # (B, S, E) -> (B, S, E)
        
        # If the model is set to return the full sequence, return the output
        if self.return_sequence:
            return out
        
        # If the model is not set to return the full sequence, return only the last output
        return out[:, -1:, ...] # (B, S, E) -> (B, 1, E)