from typing import Literal
from dataclasses import dataclass

from ..config import ModelConfig


@dataclass
class AutoRegressiveConfig(ModelConfig):
    """
    Configuration for the AutoRegressive architecture.
    
    Parameters:
    - max_sequence_length (int): The maximum length of the input sequence.
    - return_sequence (bool): Whether to return the full sequence or just the last output.
    - use_cache (bool): Whether to use caching for autoregressive generation. 
        If True, the model will feed only the last token after the first step and use start_pos for cache-aware models. 
        If False, the model will feed the full cropped window at each step.
    - input_type (Literal["discrete", "continuous"]): Type of input data. 
        If 'discrete', the model handles categorical data; if 'continuous', it handles real-valued data.
    """
    
    # Configuration parameters
    max_sequence_length: int
    return_sequence: bool = False
    use_cache: bool = False
    input_type: Literal["discrete", "continuous"] = "continuous"

