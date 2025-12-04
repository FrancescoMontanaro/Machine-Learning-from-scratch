from dataclasses import dataclass
from typing import Optional, Union

from ...auto_regressive import AutoRegressiveConfig
from .blocks.vanilla_block import TransformerBlockConfig
from .blocks.deepseek_v3_1 import DeepSeekTransformerBlockConfig


@dataclass(kw_only=True)
class TransformerConfig(AutoRegressiveConfig):
    """
    Configuration for the Transformer architecture.
    
    Parameters:
    - input_dim (int): Dimension of the input features (e.g., vocabulary size for text or number of features for tabular data).
    - embed_dim (int): Dimension of the embedding space.
    - num_blocks (int): Number of Transformer blocks.
    - positional_encoding_type (str): Type of positional encoding ("learned" or "sinusoidal").
    - block_config (Union[TransformerBlockConfig, DeepSeekTransformerBlockConfig]): Configuration for the Transformer blocks.
    - output_dim (Optional[int]): Dimension of the output features. If None, defaults to input_dim.
    """
    
    # Configuration parameters
    input_dim: int
    embed_dim: int
    num_blocks: int
    positional_encoding_type: str = "learned"
    block_config: Union[TransformerBlockConfig, DeepSeekTransformerBlockConfig]
    output_dim: Optional[int] = None # If None, defaults to input_dim
    
    def __post_init__(self):
        """
        Post-initialization processing.
        """
        
        # If output_dim is not specified, set it to input_dim
        if self.output_dim is None:
            self.output_dim = self.input_dim
            
        # If the attention mechanism is the vanilla one and the head size is not specified, set it to embed_dim // num_heads
        if isinstance(self.block_config, TransformerBlockConfig) and self.block_config.attention_config.head_size is None:
            # Check if the embedding size is divisible by the number of heads
            assert self.embed_dim % self.block_config.attention_config.num_heads == 0, f"Embedding size {self.embed_dim} must be divisible by the number of heads {self.block_config.attention_config.num_heads}."
            
            # Set the head size
            self.block_config.attention_config.head_size = self.embed_dim // self.block_config.attention_config.num_heads
            
        # If the feed-forward network is the vanilla MLP and the hidden dimension is not specified, set it to 4 * embed_dim
        if isinstance(self.block_config, TransformerBlockConfig) and self.block_config.ffn_config.hidden_dim is None:
            self.block_config.ffn_config.hidden_dim = 4 * self.embed_dim
    
        # If max_seq_len is not specified, set it to max_sequence_length
        if self.block_config.attention_config.max_seq_len is None:
            self.block_config.attention_config.max_seq_len = self.max_sequence_length
    
    
@dataclass
class EncoderDecoderTransformerConfig:
    """
    Configuration for the Transformer Encoder-Decoder architecture.
    
    Parameters:
    - encoder_config (TransformerConfig): Configuration for the encoder.
    - decoder_config (TransformerConfig): Configuration for the decoder.
    """
    
    # Configuration parameters
    encoder_config: TransformerConfig
    decoder_config: TransformerConfig