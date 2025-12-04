import numpy as np

from ....layers import Dense, Dropout
from ..config import AttentionConfig
from ....core import Tensor, Module, ModuleList
from ....core.utils.data_processing import concat


class SelfSingleHeadAttention(Module):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        config: AttentionConfig,
        *args, **kwargs
    ) -> None:
        """
        Class constructor for SingleHeadAttention layer.
        
        Parameters:
        - config (AttentionConfig): The configuration for the attention layer.
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Ensure head_size is specified in the configuration
        assert config.head_size is not None, "head_size must be specified in the configuration."
        assert config.max_seq_len is not None, "max_seq_len must be specified in the configuration."
        
        # Store the head size
        self.head_size = config.head_size
        self.causal_attention = config.causal
        self.max_seq_len = config.max_seq_len
        
        # Define the key, query and value matrices
        self.key = Dense(config.head_size, add_bias=False) # (B, S, E) -> (B, S, H)
        self.query = Dense(config.head_size, add_bias=False) # (B, S, E) -> (B, S, H)
        self.value = Dense(config.head_size, add_bias=False) # (B, S, E) -> (B, S, H)

        # Creating a dropout layer
        self.dropout = Dropout(config.dropout) # (B, S, H) -> (B, S, H)
        
        # Registering the attention mask as a buffer
        self.attention_mask: Tensor # (S, S) -> (S, S)
        
    
    ### Protected methods ###
    
    def _forward(self, x: Tensor) -> Tensor:
        """
        Foward pass of the layer
        
        Parameters:
        - x (Tensor): Input tensor
        
        Returns:
        - Tensor: Output of the layer
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Dimensions are:
        # - B: batch size
        # - S: sequence length
        # - E: embedding size (embedding dimension of the original data)
        # - H: head size (embedding dimension of the key, query and value matrices)
        
        # Apply the key, query and value matrices to the embeddings
        # It will be projected into a lower dimansional space H < E (even H << E)
        k = self.key(x) # (B, S, E) -> (B, S, H)
        q = self.query(x) # (B, S, E) -> (B, S, H)
        v = self.value(x) # (B, S, E) -> (B, S, H)
        
        # Compute the attention scores by taking the dot product of the query and key matrices
        scores: Tensor = q @ k.transpose((0, 2, 1)) * (self.head_size ** -0.5)
        
        # If causal attention is enabled, we need to apply the attention mask
        if self.causal_attention:
            # Unpack the shape of the input data for better readability
            _, S, _ = x.shape # (B, S, E)
            
            # Apply the attention mask to the scores
            scores = scores.masked_fill(self.attention_mask.data[:S, :S] == 0, float('-inf'))
            
        # Normalize the attention scores using softmax to get the attention weights
        attention_weights = scores.softmax(axis=-1)
        
        # Apply dropout to the attention weights
        attention_weights = self.dropout(attention_weights) # (B, S, S) -> (B, S, S)
        
        # Compute the contextual embeddings by applying the attention weights to the value embeddings
        contextual_embeddings = attention_weights @ v # (B, S, S) @ (B, S, H) -> (B, S, H)
        
        # Return the contextual embeddings
        return contextual_embeddings # (B, S, H)
    
    
    def _lazy_init(self, x: Tensor) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, sequence length, embedding size)
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape) == 3, f"Invalid input shape. Input must be a 3D array. The shape must be (Batch size, sequence length, embedding size). Got shape: {x.shape}"
        
        # If causal attention is enabled, we need to create the attention mask
        if self.causal_attention:
            # Unpack the shape of the input data
            _, S, _ = x.shape # (B, S, E)
            
            # Initialize the attention mask as a lower triangular matrix for causal attention
            self.register_buffer("attention_mask", Tensor(np.tril(np.ones((self.max_seq_len, self.max_seq_len))))) # (S, S) -> (S, S)
    
    
class SelfMultiHeadAttention(Module):
    
    ### Magic methods ###
    
    def __init__(
        self,
        config: AttentionConfig,
        *args, **kwargs
    ) -> None:
        """
        Initialize the MultiHeadAttention module
        
        Parameters:
        - config (AttentionConfig): The configuration for the attention layer.
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Creating the attention heads
        self.heads: ModuleList[SelfSingleHeadAttention] = ModuleList([
            SelfSingleHeadAttention(config=config) 
            for _ in range(config.num_heads)
        ]) # (B, S, E) -> (B, S, H * n_heads)
        
        # Create the output linear layer to project the embeddings back to the original size
        # This will be initialize lazily, since we do not need the embedding size (E) until 
        # the first forward pass (by design of this DL framework)
        self.output_linear: Dense # (B, S, H * n_heads) -> (B, S, E)
        
        # Create the dropout layer
        self.dropout: Dropout = Dropout(config.dropout) # (B, S, E) -> (B, S, E)


    ### Protected methods ###
    
    def _forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MultiHeadAttention module
        
        Parameters:
        - x (Tensor): The input tensor
        
        Returns:
        - Tensor: The output tensor
        """
        
        # Dimensions are:
        # - B: batch size
        # - S: sequence length
        # - E: embedding size (embedding dimension of the original data)
        # - H: head size (embedding dimension of the key, query and value matrices)
        
        # Apply each head to the embeddings
        out = concat([head(x) for head in self.heads], axis=-1) # (B, S, E) -> (B, S, H * n_heads)
        
        # Apply the output linear layer to project the embeddings back to the original size
        return self.dropout(self.output_linear(out)) # (B, S, H * n_heads) -> (B, S, E)
        
    
    def _lazy_init(self, x: Tensor) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, sequence length, embedding size)
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape) == 3, f"Invalid input shape. Input must be a 3D array. The shape must be (Batch size, sequence length, embedding size). Got shape: {x.shape}"
        
        # Unpack the shape of the input data
        _, _, E = x.shape # (B, S, E)
        
        # Initialize the output linear layer
        self.output_linear = Dense(E) # (B, S, H * n_heads) -> (B, S, E)