from typing import Optional, Literal

from .decoder import Decoder
from ..auto_regressive import AutoRegressive


class Transformer(AutoRegressive):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        input_dim: int,
        n_embed: int, 
        n_attention_heads: int, 
        sequence_length: int,
        n_decoder_blocks: int = 4, 
        dropout: float = 0.1, 
        return_sequence: bool = False,
        output_dim: Optional[int] = None,
        input_type: Literal["discrete", "continuous"] = "discrete",
        do_sample: bool = True,
        *args, **kwargs
    ) -> None:
        """
        Initialize the transformer model.
        
        Parameters:
        - input_dim (int): The input dimension. It is the vocabulary size for text data or the number of features for other types of data
        - n_embed (int): The embedding size.
        - n_attention_heads (int): The number of attention heads.
        - sequence_length (int): The sequence length of the input data.
        - n_decoder_blocks (int): The number of transformer blocks.
        - dropout (float): The dropout rate.
        - return_sequence (bool): Whether to return the sequence or just the last output.
        - output_dim (Optional[int]): The output dimension. If None, it will be set to the input dimension.
        - input_type (Literal["discrete", "continuous"]): The type of input data. It can be "discrete" for text data or "continuous" for other types of data.
        - do_sample (bool): Whether to sample from the distribution or take the argmax. Default is True.
        """
        
        # Create the decoder module
        decoder: Decoder = Decoder( # (B, S) -> (B, S, O) if return_sequence is True, else (B, O)
            input_dim = input_dim,
            n_embed = n_embed,
            n_attention_heads = n_attention_heads,
            sequence_length = sequence_length,
            n_decoder_blocks = n_decoder_blocks,
            dropout = dropout,
            return_sequence = return_sequence,
            output_dim = output_dim,
            input_type = input_type
        )
        
        # Initialize the superclass
        super().__init__(
            sequence_length = sequence_length,
            return_sequence = return_sequence,
            input_type = input_type,
            do_sample = do_sample,
            modules = [decoder], 
            *args, **kwargs
        )