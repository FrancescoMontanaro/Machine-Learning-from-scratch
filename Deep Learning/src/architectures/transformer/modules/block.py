from typing import Optional

from .mlp import MLP
from ....core import Tensor, Module
from ....layers import LayerNormalization
from .self_attention import SelfMultiHeadAttention
from .cross_attention import CrossMultiHeadAttention


class Block(Module):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        n_heads: int, 
        dropout: float = 0.1, 
        causal_attention: bool = False,
        use_cross_attention: bool = False,
        *args, **kwargs
    ) -> None:
        """
        Initialize the transformer's decoder block.
        
        Parameters:
        - n_heads (int): The number of attention heads.
        - dropout (float): The dropout rate.
        - causal_attention (bool): Whether to use causal attention (default: False).
        - use_cross_attention (bool): Whether to use cross-attention (default: False).
        """
        
        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        # Store the configuration parameters
        self.n_heads = n_heads
        self.dropout = dropout
        self.causal_attention = causal_attention
        self.use_cross_attention = use_cross_attention
        
        # Define the modules of the transformer's block
        # Some of them will be lazily initialized in the forward pass, since we do not know the embedding size yet
        
        # Define the self-attention mechanism
        self.layer_norm_1 = LayerNormalization() # (B, S, E) -> (B, S, E)
        self.self_attention_heads: SelfMultiHeadAttention # (B, S, E) -> (B, S, E)
        
        # Cross-attention components (only for decoder blocks)
        if use_cross_attention:
            # Define the cross-attention mechanism
            self.layer_norm_cross = LayerNormalization() # (B, S, E) -> (B, S, E)
            self.cross_attention_heads: CrossMultiHeadAttention # (B, S_dec, E) -> (B, S_dec, E)
        
        # Define the MLP module
        self.mlp: MLP = MLP(dropout) # (B, S, E) -> (B, S, E)
        self.layer_norm_2 = LayerNormalization() # (B, S, E) -> (B, S, E)
      
    
    ### Protected methods ###  
        
    def _forward(self, x: Tensor, encoder_output: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the transformer block.
        
        Parameters:
        - x (Tensor): The input x.
        - encoder_output (Optional[Tensor]): The output from the encoder (only for decoder blocks with cross-attention).
        
        Returns:
        - Tensor: The output embeddings.
        
        Raises:
        - ValueError: If cross-attention is used but encoder_output is not provided.
        """
        
        # Dimensions are:
        # - B: batch size
        # - S: sequence length
        # - E: embedding size (embedding dimension of the original data)
        
        # Apply the multi-attention mechanism with skip connections
        out = x + self.self_attention_heads(self.layer_norm_1(x)) # (B, S, E) + (B, S, E) -> (B, S, E)
        
        # Check if cross-attention has to be applied
        if self.use_cross_attention:
            # Ensure that encoder_output is provided
            if encoder_output is None:
                # If cross-attention is used and encoder_output is not provided, raise an error
                raise ValueError("encoder_output must be provided when use_cross_attention=True")
            
            # Apply cross-attention with residual connection
            out = out + self.cross_attention_heads(self.layer_norm_cross(out), encoder_output) # (B, S, E) + (B, S, E) -> (B, S, E)
        
        # Apply the MLP module with skip connections
        return out + self.mlp(self.layer_norm_2(out)) # (B, S, E) + (B, S, E) -> (B, S, E)
        
        
    def _lazy_init(self, x: Tensor, *args, **kwargs) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, sequence length, embedding size)
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape) == 3, f"Invalid input shape. Input must be a 3D array. The shape must be (Batch size, sequence length, embedding size). Got shape: {x.shape}"
        
        # Store the input shape of the layer
        _, _, E = x.shape # (B, S, E)
        
        # Check if the embedding size is divisible by the number of heads
        assert E % self.n_heads == 0, f"Embedding size {E} must be divisible by the number of heads {self.n_heads}."
        
        # A good head size is the embedding size divided by the number of heads
        head_size = E // self.n_heads
        
        # Initialize the multi-head attention mechanism
        self.self_attention_heads = SelfMultiHeadAttention( # (B, S, E) -> (B, S, E)
            n_heads = self.n_heads, 
            head_size = head_size, 
            dropout = self.dropout,
            causal_attention = self.causal_attention
        )
        
        # Initialize cross-attention if needed
        if self.use_cross_attention:
            # Create a non-causal cross-attention mechanism
            self.cross_attention_heads = CrossMultiHeadAttention( # (B, S_dec, E) -> (B, S_dec, E)
                n_heads = self.n_heads,
                head_size = head_size,
                dropout = self.dropout,
                causal_attention = False  # Cross-attention is not causal
            )