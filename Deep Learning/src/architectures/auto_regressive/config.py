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
    - input_type (Literal["discrete", "continuous"]): Type of input data. 
        If 'discrete', the model handles categorical data; if 'continuous', it handles real-valued data.
    """
    
    # Configuration parameters
    max_sequence_length: int
    return_sequence: bool = False
    input_type: Literal["discrete", "continuous"] = "continuous"

