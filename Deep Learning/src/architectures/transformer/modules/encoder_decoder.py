from .decoder import Decoder
from .encoder import Encoder
from ....core import Tensor, Module
from ..config import EncoderDecoderTransformerConfig


class EncoderDecoder(Module):

    ### Magic methods ###
    
    def __init__(
        self,
        config: EncoderDecoderTransformerConfig,
        *args, **kwargs,
    ) -> None:
        """
        Initialize the encoder-decoder transformer model.
        
        Parameters:
        - config (EncoderDecoderTransformerConfig): The configuration for the encoder-decoder transformer model.
        """
        
        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        # Store the configuration parameters
        self.return_sequence = config.decoder_config.return_sequence
            
        # Build the encoder
        self.encoder = Encoder(config=config.encoder_config)
        
        # Build the decoder with cross-attention enabled
        self.decoder = Decoder(config=config.decoder_config)
    
    
    ### Protected methods ###
    
    def _forward(self, x: Tensor, encoder_input: Tensor, *args, **kwargs) -> Tensor:
        """
        Full forward pass through encoder and decoder.
        
        Parameters:
        - x (Tensor): Input to the decoder        
        - encoder_input (Tensor): Input to the encoder
        
        Returns:
        - Tensor: Output from the decoder, which is the final output of the model
        """
        
        # Compute the encoder output
        encoder_output = self.encoder(x=encoder_input, *args, **kwargs).output # (B, S_enc, E) -> (B, S_enc, E)
        
        # Compute the decoder output
        decoder_output = self.decoder(x=x, encoder_output=encoder_output, *args, **kwargs).output # (B, S_dec, E) + (B, S_enc, E) -> (B, S_dec, O)

        # Check if the model is set to return the full sequence or just the last output
        if self.return_sequence:
            # If the model is set to return the full sequence, return the decoder output as is
            return decoder_output # (B, S_dec, O)
        
        # If the model is not set to return the full sequence, take only the last output
        return decoder_output[:, -1, :] # (B, S_dec, O) -> (B, 1, O)