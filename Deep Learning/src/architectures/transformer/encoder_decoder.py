from typing import Optional, Literal

from .decoder import Decoder
from .encoder import Encoder
from ...core import Tensor, Module


class EncoderDecoder(Module):

    ### Magic methods ###
    
    def __init__(
        self,
        # Encoder‑side hyper‑parameters
        encoder_input_dim: int,
        encoder_sequence_length: int,
        n_encoder_blocks: int,
        # Decoder‑side hyper‑parameters
        decoder_input_dim: int,
        decoder_sequence_length: int,
        n_decoder_blocks: int,
        # Shared hyper‑parameters
        n_embed: int,
        n_attention_heads: int,
        dropout: float = 0.1,
        # Output parameters
        return_sequence: bool = False,
        output_dim: Optional[int] = None,
        data_type: Literal["discrete", "continuous"] = "discrete",
        # Additional optional encoder/decoder parameters
        encoder_causal_attention: bool = False,
        decoder_causal_attention: bool = True,
        *args, **kwargs,
    ) -> None:
        """
        Initialize the encoder-decoder transformer model.
        
        Parameters:
        
        # Encoder parameters
        - encoder_input_dim (int): The input dimension for the encoder (vocab size or feature count)
        - encoder_sequence_length (int): The sequence length for the encoder
        - n_encoder_blocks (int): Number of encoder blocks
        - encoder_causal_attention (bool): Whether to use causal attention in the encoder. Default is False.
        
        # Decoder parameters
        - decoder_input_dim (int): The input dimension for the decoder (vocab size or feature count)
        - decoder_sequence_length (int): The sequence length for the decoder
        - n_decoder_blocks (int): Number of decoder blocks
        - decoder_causal_attention (bool): Whether to use causal attention in the decoder. Default is True.
        
        # Shared parameters
        - n_embed (int): The embedding size for both encoder and decoder
        - n_attention_heads (int): The number of attention heads for both encoder and decoder
        - dropout (float): The dropout rate for both encoder and decoder
        
        # Output parameters
        - return_sequence (bool): Whether to return the sequence or just the last output. Default is False.
        - output_dim (Optional[int]): The output dimension for the decoder. If None, it will be set to the decoder input dimension.
        - data_type (Literal["discrete", "continuous"]): The type of input data, either "discrete" for text or "continuous" for other types. Default is "discrete".
        """
        
        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        # Store the configuration parameters
        self.return_sequence = return_sequence
            
        # Build the encoder
        self.encoder = Encoder(
            input_dim = encoder_input_dim,
            n_embed = n_embed,
            n_attention_heads = n_attention_heads,
            sequence_length = encoder_sequence_length,
            n_encoder_blocks = n_encoder_blocks,
            dropout = dropout,
            input_type = data_type,
            causal_attention = encoder_causal_attention,
            return_sequence = True, # Always return the full sequence from the encoder since it is used for cross-attention in the decoder
        )
        
        # Build the decoder with cross-attention enabled
        self.decoder = Decoder(
            input_dim = decoder_input_dim,
            n_embed = n_embed,
            n_attention_heads = n_attention_heads,
            sequence_length = decoder_sequence_length,
            n_decoder_blocks = n_decoder_blocks,
            dropout = dropout,
            output_dim = output_dim,
            input_type = data_type,
            causal_attention = decoder_causal_attention,
            return_sequence = return_sequence,
            use_cross_attention = True # Enable cross-attention in the decoder to attend to the encoder's output
        )
    
    
    ### Protected methods ###
    
    def _forward(self, encoder_input: Tensor, decoder_input: Tensor) -> Tensor:
        """
        Full forward pass through encoder and decoder.
        
        Parameters:
        - encoder_input (Tensor): Input to the encoder
        - decoder_input (Tensor): Input to the decoder
        
        Returns:
        - Tensor: Output from the decoder, which is the final output of the model
        """
        
        # Compute the encoder output
        encoder_output = self.encoder(encoder_input) # (B, S_enc, E) -> (B, S_enc, E)
        
        # Compute the decoder output
        decoder_output = self.decoder(decoder_input, encoder_output) # (B, S_dec, E) -> (B, S_dec, O)

        # Check if the model is set to return the full sequence or just the last output
        if self.return_sequence:
            # If the model is set to return the full sequence, return the decoder output as is
            return decoder_output # (B, S_dec, O)
        
        # If the model is not set to return the full sequence, take only the last output
        return decoder_output[:, -1, :] # (B, S_dec, O) -> (B, 1, O)