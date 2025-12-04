from typing import Optional
from dataclasses import dataclass

from .common import BaseAttentionConfig, MLPConfig


@dataclass(kw_only=True)
class AttentionConfig(BaseAttentionConfig):
    """
    Configuration for the attention mechanism.
    
    Parameters:
    - head_size (Optional[int]): Size of each attention head. 
        If None, defaults to embed_dim // num_heads by the Transformer config.
    """
    
    # Configuration parameters
    head_size: Optional[int] = None


@dataclass
class TransformerBlockConfig:
    """
    Configuration for a Transformer block.
    
    Parameters:
    - attention_config (AttentionConfig): Configuration for the attention mechanism.
    - ffn_config (MLPConfig): Configuration for the feed-forward network.
    """
    
    # Configuration parameters
    attention_config: AttentionConfig
    ffn_config: MLPConfig