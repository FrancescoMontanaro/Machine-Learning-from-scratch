import numpy as np

from .decoder_block import DecoderBlock
from ...core import Tensor, Module, ModuleList
from ...layers import Dense, Embedding, LayerNormalization


class Decoder(Module):
    
    ### Magic methods ###
    
    def __init__(self, vocab_size: int, n_embed: int, n_attention_heads: int, sequence_length: int, n_decoder_blocks: int = 4, dropout: float = 0.1, *args, **kwargs) -> None:
        """
        Initialize the transformer's decoder.
        
        Parameters:
        - vocab_size (int): The size of the vocabulary.
        - n_embed (int): The embedding size.
        - n_attention_heads (int): The number of attention heads.
        - sequence_length (int): The sequence length of the input data.
        - n_decoder_blocks (int): The number of transformer blocks.
        - dropout (float): The dropout rate.
        """
        
        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        # Store the parameters
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_attention_heads = n_attention_heads
        self.sequence_length = sequence_length
        self.n_decoder_blocks = n_decoder_blocks
        self.dropout = dropout
        
        # Define the dimensions of the input data
        
        # Define the embedding layer
        self.embedding: Embedding = Embedding(vocab_size, n_embed) # (B, S) -> (B, S, E)
        self.positional_embedding: Embedding = Embedding(sequence_length, n_embed) # (S) -> (S, E)
        
        # Instantiate the decoder blocks
        self.decoder_blocks: ModuleList[DecoderBlock] = ModuleList([ # (B, S, E) -> (B, S, E)
            DecoderBlock(n_attention_heads, dropout) 
            for _ in range(n_decoder_blocks)
        ])
        
        # Layer normalization
        self.layer_norm: LayerNormalization = LayerNormalization() # (B, S, E) -> (B, S, E)
        
        # Define the output layer
        self.output_layer: Dense = Dense(vocab_size) # (B, S, E) -> (B, S, V)
        
        
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
        # - V: vocabulary size
        
        # Unpack the shape of the input data for better readability
        _, S = x.shape() # (B, S)
            
        # Token embeddings
        token_embeddings = self.embedding(x) # (B, S) -> (B, S, E)
        
        # Add positional encoding
        positions = self.positional_embedding(Tensor(np.arange(S))) # (S) -> (1, S) -> (1, S, E)
        
        # Embed the input data and add the positional encoding
        embeddings = token_embeddings + positions # (B, S, E) + (S, E) -> (B, S, E)
        
        # Apply the decoder blocks
        for block in self.decoder_blocks:
            embeddings = block(embeddings) # (B, S, E) -> (B, S, E)
            
        # Apply the output layer to get the logits
        return self.output_layer(self.layer_norm(embeddings)) # (B, S, E) -> (B, S, V)
    
    
    def _lazy_init(self, x: Tensor) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, sequence length, embedding size)
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape()) == 2, f"Invalid input shape. Input must be a 2D array. The shape must be (Batch size, sequence length). Got shape: {x.shape()}"