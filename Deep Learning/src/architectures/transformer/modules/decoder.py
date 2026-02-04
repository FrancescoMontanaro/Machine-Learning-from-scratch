import numpy as np
from typing import Optional

from .block import Block
from ..config import TransformerConfig
from ....core import Tensor, SingleOutputModule, ModuleList
from ....layers import Dense, Embedding, LayerNormalization, PositionalEncoding


class Decoder(SingleOutputModule):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        config: TransformerConfig,
        *args, **kwargs
    ) -> None:
        """
        Initialize the transformer's decoder.
        
        Parameters:
        - config (TransformerConfig): The configuration for the decoder transformer model.
        """
        
        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        # Store the parameters
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
        
        # Instantiate the decoder blocks
        self.decoder_blocks: ModuleList[Block] = ModuleList([ # (B, S, E) -> (B, S, E)
            Block(config=config.block_config)
            for _ in range(config.num_blocks)
        ])
        
        # Layer normalization
        self.layer_norm: LayerNormalization = LayerNormalization() # (B, S, E) -> (B, S, E)
        
        # Define the output layer 
        # If output_dim is None, defaults to input_dim
        assert config.output_dim is not None, "output_dim must be specified in the configuration."
        self.output_layer: Dense = Dense(config.output_dim) # (B, S, E) -> (B, S, O)
        
        
    ### Protected methods ###
    
    def _forward(
        self, 
        x: Tensor, 
        encoder_output: Optional[Tensor] = None,
        start_pos: int = 0
    ) -> Tensor:
        """
        Forward pass of the transformer's decoder.
        
        Parameters:
        - x (Tensor): The input x.
        - encoder_output (Optional[Tensor]): The output from the encoder (for cross-attention).
        - start_pos (int): Starting position for attention caching (used in DeepSeek blocks).
        - freq_cis (Optional[Tensor]): Precomputed RoPE frequencies (used in DeepSeek blocks).
        
        Returns:
        - Tensor: The output logits.
        
        Raises:
        - ValueError: If cross-attention is used but encoder_output is not provided.
        """
        
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
            positions = Tensor(np.arange(start_pos, start_pos + S)) # Create position indices (S,)
            pos_embeddings = self.positional_encoding(positions) # (S,) -> (S, E)
            
            # Add positional embeddings to input embeddings
            embeddings = embeddings + pos_embeddings # (B, S, E) + (S, E) -> (B, S, E)
        
        # Apply the decoder blocks
        for block in self.decoder_blocks:
            embeddings = block(  # (B, S, E) -> (B, S, E)
                x = embeddings, 
                start_pos = start_pos, 
                encoder_output = encoder_output
            )
            
        # Apply the output layer to get the logits
        out = self.output_layer(self.layer_norm(embeddings)) # (B, S, E) -> (B, S, O)
        
        # If the model is set to return the full sequence, return the output
        if self.return_sequence:
            return out
        
        # If the model is not set to return the full sequence, return only the last output
        return out[:, -1:, ...] # (B, S, O) -> (B, 1, O)