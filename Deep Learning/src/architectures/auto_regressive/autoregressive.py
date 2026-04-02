import numpy as np
from typing import Generator, Union, Optional, Callable

from ...core import Tensor
from ..sequential import Sequential
from .config import AutoRegressiveConfig
from ...core.utils.data_processing import concat
from ...core.utils.context_manager import no_grad


class AutoRegressive(Sequential):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        config: AutoRegressiveConfig,
        *args, **kwargs
    ) -> None:
        """
        Initialize the autoregressive architecture.
        
        Parameters:
        - config (AutoRegressiveConfig): Configuration for the autoregressive architecture.
        """
            
        # Initialize the autoregressive architecture with a sequence length.
        super().__init__(*args, **kwargs)
        
        # Save the configuration parameters
        self.max_sequence_length = config.max_sequence_length
        self.return_sequence = config.return_sequence
        self.input_type = config.input_type
        self.use_cache = config.use_cache
    
    
    ### Public Methods ###
    
    def autoregressive_generation(
        self, 
        x: Tensor, 
        num_steps: int, 
        concat_axis: int = 1, 
        stream: bool = False,
        do_sample: bool = False,
        use_cache: Optional[bool] = None,
        preprocess_fn: Optional[Callable[[Tensor], Tensor]] = None,
        postprocess_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *args, **kwargs
    ) -> Union[Tensor, Generator[Tensor, None, None]]:
        """
        Autoregressive generation function to generate data.
        
        Parameters:
        - x (Tensor): The input tensor.
        - num_steps (int): The number of steps to generate.
        - concat_axis (int): The axis to concatenate the generated data.
        - stream (bool): Whether to generate the data in a streaming fashion.
        - do_sample (bool, optional): Whether to use sampling during generation.
        - use_cache (bool, optional): If True, feed only the last token after the first step
            and use start_pos for cache-aware models. If False, feed the full cropped window at
            every step (default for recurrent models without cache).
        - preprocess_fn (Callable, optional): Function to normalize input before model forward pass.
        - postprocess_fn (Callable, optional): Function to denormalize model output.
        """
        
        # Disable gradient computation
        with no_grad():
            # Set the model to evaluation mode
            self.eval()

            # Reset the cache
            self.reset_cache()

            # Stream requested
            if stream:
                # Return the generator to stream the data
                return self._autoregressive_step_loop(
                    x = x, 
                    num_steps = num_steps, 
                    concat_axis = concat_axis, 
                    do_sample = do_sample,
                    use_cache = use_cache,
                    preprocess_fn = preprocess_fn, 
                    postprocess_fn = postprocess_fn
                )
            
            # Generate all the data at once
            else:   
                # Return the generated data
                return self.concat_generation(
                    self._autoregressive_step_loop(
                        x = x, 
                        num_steps = num_steps, 
                        concat_axis = concat_axis, 
                        do_sample = do_sample,
                        use_cache = use_cache,
                        preprocess_fn = preprocess_fn, 
                        postprocess_fn = postprocess_fn
                    ),
                    concat_axis = concat_axis
                )

            
    @staticmethod
    def concat_generation(generator: Generator[Tensor, None, None], concat_axis: int = 1) -> Tensor:
        """
        Concatenate generation function to generate data.
        
        Parameters:
        - generator (Generator): The generator yielding tensors.
        - concat_axis (int): The axis to concatenate the generated data.
        
        Returns:
        - Tensor: The concatenated generated data.
        """
        
        # Initialize the output tensor with the first element of the generator
        out = next(generator)
        
        # Iterate over the generator
        for t in generator:
            # Concatenate the generated data
            out = concat([out, t], axis=concat_axis)
            
        # Return the concatenated generated data
        return out
               
    
    ### Protected Methods ###
            
    def _autoregressive_step_loop(
        self, 
        x: Tensor, 
        num_steps: int, 
        concat_axis: int = 1,
        do_sample: bool = False,
        use_cache: Optional[bool] = None,
        preprocess_fn: Optional[Callable[[Tensor], Tensor]] = None,
        postprocess_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *args, **kwargs
    ) -> Generator[Tensor, None, None]:
        """
        Autoregressive step loop to generate data.
        
        Parameters:
        - x (Tensor): The input tensor.
        - num_steps (int): The number of steps to generate.
        - concat_axis (int): The axis to concatenate the generated data.
        - do_sample (bool, optional): Whether to use sampling during generation.
        - use_cache (bool, optional): If True, feed only the last token after step 0.
            If False, feed the full cropped context window each step.
        - preprocess_fn (Callable, optional): Function to normalize input before model forward pass.
        - postprocess_fn (Callable, optional): Function to denormalize model output.
        
        Yields:
        - Tensor: The generated data at each step.
        """
        
        # If not provided, default to the instance flag when available.
        # Fallback to False to avoid cache-specific behavior on models without cache.
        if use_cache is None:
            use_cache = self.use_cache

        # Iterate over the maximum number of new steps to generate
        for step in range(num_steps):
            # Crop the input sequence to the sequence length if larger
            cropped_x = x[:, -self.max_sequence_length:, ...]
            
            # Apply normalization if provided
            if preprocess_fn is not None:
                cropped_x = preprocess_fn(cropped_x)
                
            # Compute model input for the current step.
            if use_cache and step > 0:
                # Cache-aware path: feed only the newest token.
                start_pos = min(cropped_x.shape[1] - 1, self.max_sequence_length - 1)
                model_input = cropped_x[:, -1:, ...]
            else:
                # Non-cache path: feed the full context window every step.
                start_pos = 0
                model_input = cropped_x

            # Get the prediction from the model
            result = self(x=model_input, start_pos=start_pos)
            out = result.output

            # Apply denormalization if provided
            if postprocess_fn is not None:
                out = postprocess_fn(out)
                
            # Unsqueeze the output to add the time dimension if needed
            if out.data.ndim <= 2:
                # Add a new axis for the time dimension
                out = out.unsqueeze(axis=1)
            
            # For autoregressive generation, we always need only the last token's prediction
            # regardless of return_sequence setting
            out = out[:, -1:, ...]
            
            # If the input type is discrete, apply softmax and sample or take argmax
            if self.input_type == "discrete":
                # Apply the softmax function to get the probabilities
                out = out.softmax(axis=-1)
                
                # If sampling is enabled, sample from the distribution or take the argmax
                if do_sample:
                    # Sample the next item from the distribution
                    out = Tensor(
                        np.array([
                            np.random.choice(out.shape[-1], p=out.data[i, -1])
                            for i in range(out.shape[0])
                        ]).reshape(-1, 1),
                        requires_grad = out.requires_grad,
                        dtype = np.int32
                    )
                else:
                    # Take the argmax of the probabilities to get the next item
                    out = Tensor(
                        np.argmax(out.data, axis=-1).reshape(-1, 1),
                        requires_grad = out.requires_grad,
                        dtype = np.int32
                    )
            
            # Yield the next item
            yield out
            
            # Concatenate the logits to the input tensor along the specified axis
            x = concat([x, out], axis=concat_axis)