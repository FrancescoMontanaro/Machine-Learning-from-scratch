import numpy as np
from typing import Optional

from ...src.core.utils.data_processing import concat
from ...src import Tensor, Module, ModuleList, layers


class SingleHeadAttention(Module):
    
    ### Magic methods ###
    
    def __init__(self, head_size: int, dropout: float = 0.1, name: Optional[str] = None) -> None:
        """
        Class constructor for SingleHeadAttention layer.
        
        Parameters:
        - head_size (int): size of the attention head. The dimension of the key, query and value matrices (H)
        - dropout (float): dropout rate to be applied to the attention scores
        - name (str): name of the layer
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Store the head size
        self.head_size = head_size
        
        # Define the key, query and value matrices
        self.key = layers.Dense(head_size, add_bias=False) # (B, S, E) -> (B, S, H)
        self.query = layers.Dense(head_size, add_bias=False) # (B, S, E) -> (B, S, H)
        self.value = layers.Dense(head_size, add_bias=False) # (B, S, E) -> (B, S, H)
        
        # Creating a dropout layer
        self.dropout = layers.Dropout(dropout) # (B, S, H) -> (B, S, H)
        
        # Registering the attention mask as a buffer
        self.attention_mask: Tensor # (S, S) -> (S, S)
        
    
    ### Public methods ###
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Foward pass of the layer
        
        Parameters:
        - x (Tensor): Input tensor
        
        Returns:
        - Tensor: Output of the layer
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape()) == 3, f"Invalid input shape. Input must be a 3D array. The shape must be (Batch size, sequence length, embedding size). Got shape: {x.shape()}"
        
        # Dimensions are:
        # - B: batch size
        # - S: sequence length
        # - E: embedding size (embedding dimension of the original data)
        # - H: head size (embedding dimension of the key, query and value matrices)
        
        # Store the input shape of the layer
        self.input_shape = x.shape()
        
        # Unpack the shape of the input data for better readability
        B, S, E = self.input_shape 
        
        # Check if the layer is initialized
        if not self.initialized:
            # Initialize the layer
            self.init_params(S)
        
        # Apply the key, query and value matrices to the embeddings
        # It will be projected into a lower dimansional space H < E (even H << E)
        k = self.key(x) # (B, S, E) -> (B, S, H)
        q = self.query(x) # (B, S, E) -> (B, S, H)
        v = self.value(x) # (B, S, E) -> (B, S, H)
        
        # Compute the attention weights, apply the attention mask and normalize the weights using softmax
        attention_weights: Tensor = k @ q.transpose((-2, -1)) * self.head_size ** -0.5 # (B, S, H) @ (B, H, S) -> (B, S, S)
        attention_weights = attention_weights.masked_fill(self.attention_mask.data[:S, :S] == 0, float('-inf')) # (B, S, S) -> (B, S, S)
        attention_weights = attention_weights.softmax(axis=-1) # (B, S, S) -> (B, S, S)
        
        # Apply dropout to the attention weights
        attention_weights = self.dropout(attention_weights) # (B, S, S) -> (B, S, S)
        
        # Compute the contextual embeddings by applying the attention weights to the value embeddings
        contextual_embeddings = attention_weights @ v # (B, S, S) @ (B, S, H) -> (B, S, H)
        
        # Return the contextual embeddings
        return contextual_embeddings # (B, S, H)
    
    
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tuple: The shape of the output of the layer
        """
        
        # Call the parent class method to check if the layer is initialized
        super().output_shape()
        
        # Unpack the input shape for better readability
        B, S, H = self.input_shape
        
        # The output shape
        return (B, S, H) # (B, S, H)
    
    
    def init_params(self, S: int) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - S (int): The sequence length of the input data
        """
        
        # Initialize the attention mask as a lower triangular matrix for causal attention
        self.register_buffer("attention_mask", Tensor(np.tril(np.ones((S, S))))) # (S, S) -> (S, S)
        
        # Call the parent class method to set the layer as initialized
        super().init_params()
    
    
    
class MultiHeadAttention(Module):
    
    ### Magic methods ###
    
    def __init__(self, n_heads: int, head_size: int, dropout : float = 0.1, name: Optional[str] = None) -> None:
        """
        Initialize the MultiHeadAttention module
        
        Parameters:
        - n_heads (int): The number of attention heads.
        - head_size (int): The size of each attention head (H).
        - dropout (float): The dropout rate to apply to the attention scores.
        - name (str): The name of the module.
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Creating the attention heads
        self.heads = ModuleList([SingleHeadAttention(head_size) for _ in range(n_heads)]) # (B, S, E) -> (B, S, H * n_heads)
        
        # Create the output linear layer to project the embeddings back to the original size
        # This will be initialize lazily, since we do not need the embedding size (E) until 
        # the first forward pass (by design of this DL framework)
        self.output_linear: Module # (B, S, H * n_heads) -> (B, S, E)
        
        # Create the dropout layer
        self.dropout = layers.Dropout(dropout) # (B, S, E) -> (B, S, E)
        
        
    ### Public methods ###
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MultiHeadAttention module
        
        Parameters:
        - x (Tensor): The input tensor
        
        Returns:
        - Tensor: The output tensor
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape()) == 3, f"Invalid input shape. Input must be a 3D array. The shape must be (Batch size, sequence length, embedding size). Got shape: {x.shape()}"
        
        # Dimensions are:
        # - B: batch size
        # - S: sequence length
        # - E: embedding size (embedding dimension of the original data)
        # - H: head size (embedding dimension of the key, query and value matrices)
        
        # Store the input shape of the layer
        self.input_shape = x.shape()
        
        # Unpack the shape of the input data for better readability
        B, S, E = self.input_shape
        
        # Check if the layer is initialized
        if not self.initialized:
            # Initialize the layer
            self.init_params(E)
        
        # Apply each head to the embeddings
        out = concat([head(x) for head in self.heads], axis=-1) # (B, S, E) -> (B, S, H * n_heads)
        
        # Apply the output linear layer to project the embeddings back to the original size
        return self.dropout(self.output_linear(out)) # (B, S, H * n_heads) -> (B, S, E)
    
    
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the module
        
        Returns:
        - tuple: The shape of the output of the module
        """
        
        # Call the parent class method to check if the module is initialized
        super().output_shape()
        
        # Return the output shape
        return self.input_shape # (B, S, E)
    
    
    def init_params(self, E: int) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - E (int): The embedding size of the input data
        """
        
        # Initialize the output linear layer
        self.output_linear = layers.Dense(E) # (B, S, H * n_heads) -> (B, S, E)
        
        # Call the parent class method to set the layer as initialized
        super().init_params()