import numpy as np

from ..config import AttentionConfig
from ....layers import Dense, Dropout
from ....core import Tensor, SingleOutputModule, ModuleList
from ....core.utils.data_processing import concat


class CrossSingleHeadAttention(SingleOutputModule):
    
    ### Magic methods ###
    
    def __init__(
        self,
        config: AttentionConfig,
        *args, **kwargs
    ) -> None:
        """
        Class constructor for CrossSingleHeadAttention layer.
        
        Parameters:
        - config (AttentionConfig): The configuration for the attention layer.
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Ensure head_size is specified in the configuration
        assert config.head_size is not None, "head_size must be specified in the configuration."
        
        # Store the head size
        self.head_size = config.head_size
        self.causal_attention = False  # Cross-attention is not causal
        
        # Define the key, query and value matrices
        self.key = Dense(config.head_size, add_bias=False) # (B, S_enc, E) -> (B, S_enc, H)
        self.query = Dense(config.head_size, add_bias=False) # (B, S_dec, E) -> (B, S_dec, H)
        self.value = Dense(config.head_size, add_bias=False) # (B, S_enc, E) -> (B, S_enc, H)
        
        # Creating a dropout layer
        self.dropout = Dropout(config.dropout) # (B, S_dec, H) -> (B, S_dec, H)
        
        # Registering the attention mask as a buffer
        self.attention_mask: Tensor # (S_enc, S_dec) -> (S_enc, S_dec)
        
    
    ### Protected methods ###
    
    def _forward(self, decoder_x: Tensor, encoder_output: Tensor) -> Tensor:
        """
        Forward pass of the cross-attention layer
        
        Parameters:
        - decoder_x (Tensor): Decoder input tensor (queries come from decoder)
        - encoder_output (Tensor): Encoder output tensor (keys and values come from encoder)
        
        Returns:
        - Tensor: Output of the cross-attention layer
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Dimensions are:
        # - B: batch size
        # - S_dec: decoder sequence length
        # - S_enc: encoder sequence length
        # - E: embedding size (embedding dimension of the original data)
        # - H: head size (embedding dimension of the key, query and value matrices)
        
        # Apply the key, query and value matrices
        # Query comes from decoder, Key and Value come from encoder
        q = self.query(decoder_x) # (B, S_dec, E) -> (B, S_dec, H)
        k = self.key(encoder_output) # (B, S_enc, E) -> (B, S_enc, H)
        v = self.value(encoder_output) # (B, S_enc, E) -> (B, S_enc, H)
        
        # Compute cross-attention scores
        scores: Tensor = q @ k.transpose((0, 2, 1)) * (self.head_size ** -0.5) # (B, S_dec, H) @ (B, H, S_enc) -> (B, S_dec, S_enc)
        
        # If causal attention is enabled, we need to apply the attention mask
        if self.causal_attention:
            # Unpack the shape of the input data for better readability
            _, S_dec, _ = decoder_x.shape # (B, S_dec, E)
            _, S_enc, _ = encoder_output.shape # (B, S_enc, E)
            
            # Apply the attention mask to the scores
            scores = scores.masked_fill(self.attention_mask.data[:S_dec, :S_enc] == 0, float('-inf'))
        
        # Compute the attention weights by applying softmax to the scores
        attention_weights = scores.softmax(axis=-1)
        
        # Apply dropout to the attention weights
        attention_weights = self.dropout(attention_weights) # (B, S_dec, S_enc) -> (B, S_dec, S_enc)
        
        # Compute the contextual embeddings by applying the attention weights to the value embeddings
        contextual_embeddings = attention_weights @ v # (B, S_dec, S_enc) @ (B, S_enc, H) -> (B, S_dec, H)
        
        # Return the contextual embeddings
        return contextual_embeddings # (B, S_dec, H)
    
    
    def _lazy_init(self, decoder_x: Tensor, encoder_output: Tensor) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - decoder_x (Tensor): Decoder input data. Shape: (Batch size, decoder sequence length, embedding size)
        - encoder_output (Tensor): Encoder output data. Shape: (Batch size, encoder sequence length, embedding size)
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shapes are valid
        assert len(decoder_x.shape) == 3, f"Invalid decoder input shape. Input must be a 3D array. Got shape: {decoder_x.shape}"
        assert len(encoder_output.shape) == 3, f"Invalid encoder output shape. Input must be a 3D array. Got shape: {encoder_output.shape}"
        
        # If causal attention is enabled, we need to create the attention mask
        if self.causal_attention:
            # Unpack the shape of the input data
            _, S_dec, _ = decoder_x.shape # (B, S_dec, E)
            _, S_enc, _ = encoder_output.shape # (B, S_enc, E)
            
            # Initialize the attention mask as a lower triangular matrix for causal attention
            self.register_buffer("attention_mask", Tensor(np.tril(np.ones((S_enc, S_dec))))) # (S, S) -> (S, S)


class CrossMultiHeadAttention(SingleOutputModule):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        config: AttentionConfig,
        *args, **kwargs
    ) -> None:
        """
        Initialize the CrossMultiHeadAttention module
        
        Parameters:
        - config (AttentionConfig): The configuration for the attention layer.
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Creating the cross-attention heads
        self.heads: ModuleList[CrossSingleHeadAttention] = ModuleList([
            CrossSingleHeadAttention(config=config) 
            for _ in range(config.num_heads)
        ])
        
        # Create the output linear layer to project the embeddings back to the original size
        self.output_linear: Dense # (B, S_dec, H * n_heads) -> (B, S_dec, E)
        
        # Create the dropout layer
        self.dropout: Dropout = Dropout(config.dropout) # (B, S_dec, E) -> (B, S_dec, E)
        
        
    ### Protected methods ###
    
    def _forward(self, decoder_x: Tensor, encoder_output: Tensor) -> Tensor:
        """
        Forward pass of the CrossMultiHeadAttention module
        
        Parameters:
        - decoder_x (Tensor): The decoder input tensor
        - encoder_output (Tensor): The encoder output tensor
        
        Returns:
        - Tensor: The output tensor
        """
        
        # Apply each head to the embeddings
        out = concat([head(decoder_x, encoder_output) for head in self.heads], axis=-1) # (B, S_dec, E) -> (B, S_dec, H * n_heads)
        
        # Apply the output linear layer to project the embeddings back to the original size
        return self.dropout(self.output_linear(out)) # (B, S_dec, H * n_heads) -> (B, S_dec, E)
        
    
    def _lazy_init(self, decoder_x: Tensor, encoder_output: Tensor) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - decoder_x (Tensor): Decoder input data. Shape: (Batch size, decoder sequence length, embedding size)
        - encoder_output (Tensor): Encoder output data. Shape: (Batch size, encoder sequence length, embedding size)
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shapes are valid
        assert len(decoder_x.shape) == 3, f"Invalid decoder input shape. Got shape: {decoder_x.shape}"
        assert len(encoder_output.shape) == 3, f"Invalid encoder output shape. Got shape: {encoder_output.shape}"
        
        # Unpack the shape of the decoder input data
        _, _, E = decoder_x.shape # (B, S_dec, E)
        
        # Initialize the output linear layer
        self.output_linear = Dense(E) # (B, S_dec, H * n_heads) -> (B, S_dec, E)