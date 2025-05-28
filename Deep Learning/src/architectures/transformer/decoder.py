import numpy as np
from typing import Optional, Literal

from .decoder_block import DecoderBlock
from ...core import Tensor, Module, ModuleList
from ...layers import Dense, Embedding, LayerNormalization


class Decoder(Module):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        input_dim: int, 
        n_embed: int, 
        n_attention_heads: int, 
        sequence_length: int, 
        n_decoder_blocks: int = 4, 
        dropout: float = 0.1, 
        return_sequence: bool = False,
        output_dim: Optional[int] = None,
        input_type: Literal["discrete", "continuous"] = "discrete",
        *args, **kwargs) -> None:
        """
        Initialize the transformer's decoder.
        
        Parameters:
        - input_dim (int): The input dimension. It is the vocabulary size for text data or the number of features for other types of data.
        - n_embed (int): The embedding size.
        - n_attention_heads (int): The number of attention heads.
        - sequence_length (int): The sequence length of the input data.
        - n_decoder_blocks (int): The number of transformer blocks.
        - dropout (float): The dropout rate.
        - return_sequence (bool): Whether to return the sequence or just the last output.
        - output_dim (Optional[int]): The output dimension. If None, it will be set to the input dimension.
        - input_type (Literal["discrete", "continuous"]): The type of input data. It can be "discrete" for text data or "continuous" for other types of data.
        """
        
        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        # Store the parameters
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.n_embed = n_embed
        self.n_attention_heads = n_attention_heads
        self.sequence_length = sequence_length
        self.n_decoder_blocks = n_decoder_blocks
        self.dropout = dropout
        self.return_sequence = return_sequence
        
        # Define the input projection layer
        if input_type == "discrete":
            self.input_proj = Embedding(input_dim, n_embed) # (B, S) -> (B, S, E)
        else:
            self.input_proj = Dense(n_embed) # (B, S, F) -> (B, S, E)
        
        # Define the positional embedding layer
        self.positional_embedding: Embedding = Embedding(sequence_length, n_embed) # (S) -> (S, E)
        
        # Instantiate the decoder blocks
        self.decoder_blocks: ModuleList[DecoderBlock] = ModuleList([ # (B, S, E) -> (B, S, E)
            DecoderBlock(n_attention_heads, dropout) 
            for _ in range(n_decoder_blocks)
        ])
        
        # Layer normalization
        self.layer_norm: LayerNormalization = LayerNormalization() # (B, S, E) -> (B, S, E)
        
        # Define the output layer
        self.output_layer: Dense = Dense(self.output_dim) # (B, S, E) -> (B, S, O)
        
        
    ### Protected methods ###
    
    def _forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the transformer's decoder.
        
        Parameters:
        - x (Tensor): The input x.
        
        Returns:
        - Tensor: The output logits.
        """
        
        # Dimensions are:
        # - B: batch size
        # - S: sequence length
        # - E: embedding size (the dimension of the embedding space)
        # - O: output dimension (the dimension of the output space, e.g., vocabulary size for text data)
        
        # Unpack the shape of the input data for better readability
        _, S, *_ = x.shape() # (B, S)
            
        # Project the input data to the embedding space
        embeddings = self.input_proj(x) # (B, S) -> (B, S, E)
        
        # Add positional encoding
        positions = self.positional_embedding(Tensor(np.arange(S))) # (S) -> (1, S) -> (1, S, E)
        
        # Embed the input data and add the positional encoding
        embeddings = embeddings + positions # (B, S, E) + (S, E) -> (B, S, E)
        
        # Apply the decoder blocks
        for block in self.decoder_blocks:
            embeddings = block(embeddings) # (B, S, E) -> (B, S, E)
            
        # Apply the output layer to get the logits
        out = self.output_layer(self.layer_norm(embeddings)) # (B, S, E) -> (B, S, O)
        
        # If return_sequence is True, return the entire sequence
        if self.return_sequence:
            return out
        
        # Return the logits for the last item in the sequence
        return out[:, -1, :] 