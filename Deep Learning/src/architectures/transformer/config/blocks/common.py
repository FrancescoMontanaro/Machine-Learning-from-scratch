from typing import Optional
from dataclasses import dataclass


@dataclass
class MLPConfig:
    """
    Configuration for the vanilla feed-forward network (MLP).
    
    Parameters:
    - dropout (float): Dropout rate.
    - hidden_dim (Optional[int]): Dimension of the hidden layer. 
        If None, defaults to 4 times the input dimension by the Transformer config.
    """
    
    # Configuration parameters
    dropout: float = 0.1
    hidden_dim: Optional[int] = None


@dataclass
class BaseAttentionConfig:
    """
    Base configuration for attention mechanisms.
    
    Parameters:
    - num_heads (int): Number of attention heads.
    - dropout (float): Dropout rate.
    - causal (bool): Whether to use causal attention.
    - max_seq_len (Optional[int]): Maximum sequence length. If None, and this config is used in a TransformerBlockConfig, it defaults to the transformer's max_seq_len.
    """
    
    num_heads: int
    dropout: float
    causal: bool = True
    max_seq_len: Optional[int] = None
