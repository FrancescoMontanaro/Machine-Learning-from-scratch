import numpy as np
from typing import Generator, Union, Optional, Callable, Literal

from ...core import Tensor
from ..auto_regressive import AutoRegressive
from ...core.utils.data_processing import concat
from ...core.utils.context_manager import no_grad

# Import transformer modules
from .modules.decoder import Decoder
from .modules.encoder import Encoder
from .modules.encoder_decoder import EncoderDecoder


class EncoderTransformer(AutoRegressive):
    
    ### Magic methods ###

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        n_encoder_blocks: int,
        n_embed: int,
        n_attention_heads: int,
        dropout: float = 0.1,
        return_sequence: bool = False,
        data_type: Literal["discrete", "continuous"] = "discrete",
        positional_encoding_type: Literal["fixed", "learned"] = "learned",
        causal_attention: bool = False,
        do_sample: bool = True,
        *args, **kwargs
    ) -> None:
        """
        Initialize the encoder transformer model.
        
        Parameters:
        - input_dim (int): The input dimension (vocab size or feature count)
        - sequence_length (int): The sequence length
        - n_encoder_blocks (int): Number of encoder blocks
        - n_embed (int): The embedding size
        - n_attention_heads (int): The number of attention heads
        - dropout (float): The dropout rate
        - return_sequence (bool): Whether to return the sequence or just the last output
        - data_type (Literal["discrete", "continuous"]): The type of input data, either "discrete" for text or "continuous" for other types
        - positional_encoding_type (Literal["fixed", "learned"]): The type of positional encoding to use. Default is "learned".
        - causal_attention (bool): Whether to use causal attention in the encoder. Default is False.
        - do_sample (bool): Whether to sample from the distribution or take the argmax. Default is True.
        """

        # Create the encoder
        encoder = Encoder(
            input_dim = input_dim,
            n_embed = n_embed,
            n_attention_heads = n_attention_heads,
            sequence_length = sequence_length,
            n_encoder_blocks = n_encoder_blocks,
            dropout = dropout,
            input_type = data_type,
            causal_attention = causal_attention,
            return_sequence = return_sequence,
            positional_encoding_type = positional_encoding_type
        )
        
        # Initialize the superclass
        super().__init__(
            sequence_length = sequence_length,
            return_sequence = return_sequence,
            input_type = data_type,
            do_sample = do_sample,
            modules = [encoder], 
            *args, **kwargs
        )



class DecoderTransformer(AutoRegressive):
    
    ### Magic methods ###

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        n_decoder_blocks: int,
        n_embed: int,
        n_attention_heads: int,
        dropout: float = 0.1,
        return_sequence: bool = False,
        data_type: Literal["discrete", "continuous"] = "discrete",
        positional_encoding_type: Literal["fixed", "learned"] = "learned",
        causal_attention: bool = True,
        do_sample: bool = True,
        *args, **kwargs
    ) -> None:
        """
        Initialize the decoder transformer model.
        
        Parameters:
        - input_dim (int): The input dimension (vocab size or feature count)
        - sequence_length (int): The sequence length
        - n_decoder_blocks (int): Number of decoder blocks
        - n_embed (int): The embedding size
        - n_attention_heads (int): The number of attention heads
        - dropout (float): The dropout rate
        - return_sequence (bool): Whether to return the sequence or just the last output
        - data_type (Literal["discrete", "continuous"]): The type of input data, either "discrete" for text or "continuous" for other types
        - positional_encoding_type (Literal["fixed", "learned"]): The type of positional encoding to use. Default is "learned".
        - causal_attention (bool): Whether to use causal attention in the decoder. Default is True.
        - do_sample (bool): Whether to sample from the distribution or take the argmax. Default is True.
        """

        # Create the decoder
        decoder = Decoder(
            input_dim = input_dim,
            n_embed = n_embed,
            n_attention_heads = n_attention_heads,
            sequence_length = sequence_length,
            n_decoder_blocks = n_decoder_blocks,
            dropout = dropout,
            input_type = data_type,
            causal_attention = causal_attention,
            return_sequence = return_sequence,
            positional_encoding_type = positional_encoding_type
        )
        
        # Initialize the superclass
        super().__init__(
            sequence_length = sequence_length,
            return_sequence = return_sequence,
            input_type = data_type,
            do_sample = do_sample,
            modules = [decoder], 
            *args, **kwargs
        )
    
   
 
class EncoderDecoderTransformer(AutoRegressive):
    
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
        positional_encoding_type: Literal["fixed", "learned"] = "learned",
        
        # Output parameters
        return_sequence: bool = False,
        output_dim: Optional[int] = None,
        data_type: Literal["discrete", "continuous"] = "discrete",
        do_sample: bool = True,
        
        # Additional optional encoder/decoder parameters
        encoder_causal_attention: bool = False,
        decoder_causal_attention: bool = True,
        *args, **kwargs
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
        - positional_encoding_type (Literal["fixed", "learned"]): The type of positional encoding to use. If "fixed", it uses a fixed positional encoding. If "learned", it uses trainable positional embeddings.
        
        # Output parameters
        - return_sequence (bool): Whether to return the sequence or just the last output. Default is False.
        - output_dim (Optional[int]): The output dimension for the decoder. If None, it will be set to the decoder input dimension.
        - data_type (Literal["discrete", "continuous"]): The type of input data, either "discrete" for text or "continuous" for other types. Default is "discrete".
        - do_sample (bool): Whether to sample from the distribution or take the argmax. Default is True.
        """
        
        # Create the encoder-decoder model
        encoder_decoder = EncoderDecoder(
            encoder_input_dim = encoder_input_dim,
            encoder_sequence_length = encoder_sequence_length,
            n_encoder_blocks = n_encoder_blocks,
            decoder_input_dim = decoder_input_dim,
            decoder_sequence_length = decoder_sequence_length,
            n_decoder_blocks = n_decoder_blocks,
            n_embed = n_embed,
            n_attention_heads = n_attention_heads,
            dropout = dropout,
            return_sequence = return_sequence,
            output_dim = output_dim,
            data_type = data_type,
            encoder_causal_attention = encoder_causal_attention,
            decoder_causal_attention = decoder_causal_attention,
            positional_encoding_type = positional_encoding_type
        )
        
        # Initialize the superclass
        super().__init__(
            sequence_length = decoder_sequence_length,
            return_sequence = return_sequence,
            input_type = data_type,
            do_sample = do_sample,
            modules = [encoder_decoder], 
            *args, **kwargs
        )
        
        
    ### Public Methods ###
    
        
    def autoregressive_generation(
        self,
        x: Tensor,
        num_steps: int,
        concat_axis: int = 1,
        stream: bool = False,
        preprocess_fn: Optional[Callable[[Tensor], Tensor]] = None,
        postprocess_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *args,
        encoder_input: Tensor,
        **kwargs
    ) -> Union[Tensor, Generator[Tensor, None, None]]:
        """
        Autoregressive generation function to generate data using the encoder-decoder transformer model.
        
        Parameters:
        - x (Tensor): The input tensor for the decoder.
        - encoder_input (Tensor): The input tensor for the encoder.
        - num_steps (int): The number of steps to generate.
        - concat_axis (int): The axis to concatenate the generated data.
        - stream (bool): Whether to generate the data in a streaming fashion.
        - preprocess_fn (Optional[Callable[[Tensor], Tensor]]): Optional preprocessing function to apply to the decoder input.
        - postprocess_fn (Optional[Callable[[Tensor], Tensor]]): Optional postprocessing function to apply to the output.
        
        Returns:
        - Union[Tensor, Generator[Tensor, None, None]]: The generated data as a Tensor or a generator yielding Tensors.
        """
        
        # Set the model to evaluation mode
        self.eval()
        
        # Disable gradient computation
        with no_grad():
            # If streaming is requested, return a generator
            if stream:
                # Return the generator to stream the data
                return self._autoregressive_step_loop(
                    x = x,
                    num_steps = num_steps,
                    concat_axis = concat_axis,
                    preprocess_fn = preprocess_fn,
                    postprocess_fn = postprocess_fn,
                    encoder_input = encoder_input
                )
                
            # Generate all the data at once
            return super().concat_generation(
                self._autoregressive_step_loop(
                    x = x,
                    num_steps = num_steps,
                    concat_axis = concat_axis,
                    preprocess_fn = preprocess_fn,
                    postprocess_fn = postprocess_fn,
                    encoder_input = encoder_input
                ),
                concat_axis = concat_axis
            )


    ### Protected Methods ###

    def _autoregressive_step_loop(
        self,
        x: Tensor,
        num_steps: int,
        concat_axis: int = 1,
        preprocess_fn: Optional[Callable[[Tensor], Tensor]] = None,
        postprocess_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *args,
        encoder_input: Tensor,
        **kwargs
    ) -> Generator[Tensor, None, None]:
        """
        Autoregressive step loop to generate data using the encoder-decoder transformer model.
        
        Parameters:
        - x (Tensor): The input tensor for the decoder.
        - encoder_input (Tensor): The input tensor for the encoder.
        - num_steps (int): The number of steps to generate.
        - concat_axis (int): The axis to concatenate the generated data.
        - preprocess_fn (Optional[Callable[[Tensor], Tensor]]): Optional preprocessing function to apply to the decoder input.
        - postprocess_fn (Optional[Callable[[Tensor], Tensor]]): Optional postprocessing function to apply to the output.
        
        Yields:
        - Tensor: The generated output at each step.
        """
        
        # Iterate for the number of steps
        for _ in range(num_steps):
            # Keep only the last decoder_sequence_length tokens
            cropped_decoder = x[:, -self.sequence_length :, ...]

            # Optional preprocessing (e.g. normalisation / embedding lookup)
            if preprocess_fn is not None:
                cropped_decoder = preprocess_fn(cropped_decoder)

            # Forward through encoder‑decoder module
            out: Tensor = self(cropped_decoder, encoder_input)

            # Optional post‑processing (e.g. projection back to original scale)
            if postprocess_fn is not None:
                out = postprocess_fn(out)
        
            # Ensure the time‑dimension is present
            if out.data.ndim <= 2: 
                out = out.unsqueeze(axis=1)

            # Select last time‑step if model returns full sequence
            if not self.return_sequence:
                out = out[:, -1:, ...]

            # If working with discrete targets, choose next token
            if self.input_type == "discrete":
                # Apply softmax to the output logits
                out = out.softmax(axis=-1)
                
                # Sample from the distribution or take argmax
                if self.do_sample:
                    # Sample the next item from the distribution
                    next_token = np.array([
                        np.random.choice(out.shape[-1], p=out.data[i, -1])
                        for i in range(out.shape[0])
                    ]).reshape(-1, 1)
                    
                # If not sampling, take the argmax
                else:
                    # Take the argmax of the output logits
                    next_token = np.argmax(out.data, axis=-1).reshape(-1, 1)
                    
                # Create a new Tensor for the next token
                out = Tensor(
                    next_token,
                    requires_grad = out.requires_grad,
                    dtype = np.int32,
                )
                
            # Yield current step prediction
            yield out

            # Append prediction to decoder_input for the next iteration
            x = concat([x, out], axis=concat_axis)