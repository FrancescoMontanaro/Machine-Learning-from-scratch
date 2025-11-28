import numpy as np
from typing import Optional, Literal

from .block import Block
from ...core import Tensor, Module, ModuleList
from ...layers import Dense, Embedding, LayerNormalization, PositionalEncoding


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
        output_dim: Optional[int] = None,
        input_type: Literal["discrete", "continuous"] = "discrete",
        causal_attention: bool = True,
        use_cross_attention: bool = False,
        return_sequence: bool = False,
        positional_encoding_type: Literal["fixed", "learned"] = "learned",
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
        - output_dim (Optional[int]): The output dimension. If None, it will be set to the input dimension.
        - input_type (Literal["discrete", "continuous"]): The type of input data. It can be "discrete" for text data or "continuous" for other types of data.
        - causal_attention (bool): Whether to use causal attention (default: True).
        - use_cross_attention (bool): Whether to use cross-attention (default: False).
        - return_sequence (bool): Whether to return the sequence or just the last output. Default is False.
        - positional_encoding_type (Literal["fixed", "learned"]): The type of positional encoding to use. It can be "fixed" for fixed positional encoding or "learned" for trainable positional embeddings.
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
        self.causal_attention = causal_attention
        self.use_cross_attention = use_cross_attention
        self.return_sequence = return_sequence
        
        # Define the input projection layer
        if input_type == "discrete":
            self.input_proj = Embedding(input_dim, n_embed) # (B, S) -> (B, S, E)
        else:
            self.input_proj = Dense(n_embed) # (B, S, F) -> (B, S, E)
        
        # Define the positional embedding layer
        self.positional_encoding = Embedding(sequence_length, n_embed) if positional_encoding_type == "learned" else PositionalEncoding(sequence_length) # (B, S) -> (B, S, E)
        
        # Instantiate the decoder blocks
        self.decoder_blocks: ModuleList[Block] = ModuleList([ # (B, S, E) -> (B, S, E)
            Block(
                n_heads = n_attention_heads, 
                dropout = dropout,
                use_cross_attention = use_cross_attention,
                causal_attention = causal_attention # Causal attention is usually used in the decoder
            ) 
            for _ in range(n_decoder_blocks)
        ])
        
        # Layer normalization
        self.layer_norm: LayerNormalization = LayerNormalization() # (B, S, E) -> (B, S, E)
        
        # Define the output layer
        self.output_layer: Dense = Dense(self.output_dim) # (B, S, E) -> (B, S, O)
        
        
    ### Protected methods ###
    
    def _forward(self, x: Tensor, encoder_output: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the transformer's decoder.
        
        Parameters:
        - x (Tensor): The input x.
        
        Returns:
        - Tensor: The output logits.
        
        Raises:
        - ValueError: If cross-attention is used but encoder_output is not provided.
        """
        
        # Check if encoder output is provided when cross-attention is used
        if self.use_cross_attention and encoder_output is None:
            # If cross-attention is used and encoder_output is not provided, raise an error
            raise ValueError("encoder_output must be provided when use_cross_attention=True")
        
        # Dimensions are:
        # - B: batch size
        # - S: sequence length
        # - E: embedding size (the dimension of the embedding space)
        # - O: output dimension (the dimension of the output space, e.g., vocabulary size for text data)
        
        # Unpack the shape of the input data for better readability
        _, S, *_ = x.shape # (B, S) or (B, S, F)
            
        # Project the input data to the embedding space
        embeddings = self.input_proj(x) # (B, S) -> (B, S, E)
        
        # Add positional encoding
        if isinstance(self.positional_encoding, PositionalEncoding):
            # Use sinusoidal positional encoding (adds directly to embeddings)
            embeddings = self.positional_encoding(embeddings) # (B, S, E) -> (B, S, E)
        else:
            # Use trainable positional embeddings
            positions = Tensor(np.arange(S)) # Create position indices (S,)
            pos_embeddings = self.positional_encoding(positions) # (S,) -> (S, E)
            
            # Add positional embeddings to input embeddings
            embeddings = embeddings + pos_embeddings # (B, S, E) + (S, E) -> (B, S, E)
        
        # Apply the decoder blocks
        for block in self.decoder_blocks:
            embeddings = block(embeddings, encoder_output) # (B, S, E) -> (B, S, E)
            
        # Apply the output layer to get the logits
        out = self.output_layer(self.layer_norm(embeddings)) # (B, S, E) -> (B, S, O)
        
        # If the model is set to return the full sequence, return the output
        if self.return_sequence:
            return out
        
        # If the model is not set to return the full sequence, return only the last output
        return out[:, -1:, ...] # (B, S, O) -> (B, 1, O)