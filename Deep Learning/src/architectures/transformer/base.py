import numpy as np
from typing import Generator, Union, Optional, Callable

from ...core import Tensor
from ..auto_regressive import AutoRegressive
from ...core.utils.data_processing import concat
from ...core.utils.context_manager import no_grad

# Import transformer modules and configurations
from .modules.decoder import Decoder
from .modules.encoder import Encoder
from .modules.encoder_decoder import EncoderDecoder
from .config import TransformerConfig, EncoderDecoderTransformerConfig


class EncoderTransformer(AutoRegressive):
    
    ### Magic methods ###

    def __init__(
        self,
        config: TransformerConfig,
        *args, **kwargs
    ) -> None:
        """
        Initialize the encoder transformer model.
        
        Parameters:
        - config (TransformerConfig): The configuration for the encoder transformer model.
        """

        # Create the encoder
        encoder = Encoder(config=config)
        
        # Initialize the superclass
        super().__init__(
            config = config,
            modules = [encoder], 
            *args, **kwargs
        )



class DecoderTransformer(AutoRegressive):
    
    ### Magic methods ###

    def __init__(
        self,
        config: TransformerConfig,
        *args, **kwargs
    ) -> None:
        """
        Initialize the decoder transformer model.
        
        Parameters:
        - config (TransformerConfig): The configuration for the decoder transformer model.
        """

        # Create the decoder
        decoder = Decoder(config=config)
        
        # Initialize the superclass
        super().__init__(
            config = config,
            modules = [decoder], 
            *args, **kwargs
        )
   
 

class EncoderDecoderTransformer(AutoRegressive):
    
    ### Magic methods ###
    
    def __init__(
        self,
        config: EncoderDecoderTransformerConfig,
        *args, **kwargs
    ) -> None:
        """
        Initialize the encoder-decoder transformer model.
        
        Parameters:
        - config (EncoderDecoderTransformerConfig): The configuration for the encoder-decoder transformer model.
        """
        
        # Create the encoder-decoder model
        encoder_decoder = EncoderDecoder(config=config)
        
        # Initialize the superclass
        super().__init__(
            config = config.decoder_config, # Use decoder config for autoregressive properties
            modules = [encoder_decoder], 
            *args, **kwargs
        )
        
        
    ### Public Methods ###
    
        
    def autoregressive_generation(
        self,
        x: Tensor,
        num_steps: int,
        concat_axis: int = 1,
        do_sample: bool = False,
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
        - do_sample (bool, optional): Whether to use sampling during generation.
        - stream (bool): Whether to generate the data in a streaming fashion.
        - preprocess_fn (Optional[Callable[[Tensor], Tensor]]): Optional preprocessing function to apply to the decoder input.
        - postprocess_fn (Optional[Callable[[Tensor], Tensor]]): Optional postprocessing function to apply to the output.
        
        Returns:
        - Union[Tensor, Generator[Tensor, None, None]]: The generated data as a Tensor or a generator yielding Tensors.
        """
        
        # Disable gradient computation
        with no_grad():
            # Set the model to evaluation mode
            self.eval()
            
            # Reset the cache
            self.reset_cache()
            
            # If streaming is requested, return a generator
            if stream:
                # Return the generator to stream the data
                return self._autoregressive_step_loop(
                    x = x,
                    num_steps = num_steps,
                    concat_axis = concat_axis,
                    do_sample = do_sample,
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
                    do_sample = do_sample,
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
        do_sample: bool = False,
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
        - do_sample (bool, optional): Whether to use sampling during generation.
        - preprocess_fn (Optional[Callable[[Tensor], Tensor]]): Optional preprocessing function to apply to the decoder input.
        - postprocess_fn (Optional[Callable[[Tensor], Tensor]]): Optional postprocessing function to apply to the output.
        
        Yields:
        - Tensor: The generated output at each step.
        """
        
        # Iterate for the number of steps
        for step in range(num_steps):
            # Keep only the last max_sequence_length tokens
            cropped_decoder = x[:, -self.max_sequence_length :, ...]

            # Optional preprocessing (e.g. normalisation / embedding lookup)
            if preprocess_fn is not None:
                cropped_decoder = preprocess_fn(cropped_decoder)

            # Compute the start position for the current input
            start_pos = 0 if step == 0 else cropped_decoder.shape[1] - 1

            # Forward through encoder‑decoder module
            out: Tensor = self(x=cropped_decoder, encoder_input=encoder_input, start_pos=start_pos)

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
                if do_sample:
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