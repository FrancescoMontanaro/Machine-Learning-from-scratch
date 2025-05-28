import numpy as np
from typing import Generator, Union, Optional, Callable, Literal

from ...core import Tensor
from ..sequential import Sequential
from ...core.utils.data_processing import concat
from ...core.utils.context_manager import no_grad


class AutoRegressive(Sequential):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        sequence_length: int,
        return_sequence: bool = False,
        input_type: Literal["discrete", "continuous"] = "continuous",
        do_sample: bool = True,
        *args, **kwargs
    ) -> None:
        """
        Initialize the autoregressive architecture.
        
        Parameters:
        - sequence_length (int): The length of the input sequence.
        - return_sequence (bool): Whether to return the sequence or just the last output. Default is False.
        - input_type (Literal["discrete", "continuous"]): The type of input data. It can be "discrete" for text data or "continuous" for other types of data.
        - do_sample (bool): Whether to sample from the distribution or take the argmax. Default is True.
        """
            
        # Initialize the autoregressive architecture with a sequence length.
        super().__init__(*args, **kwargs)
        
        # Save the configuration parameters
        self.sequence_length = sequence_length
        self.return_sequence = return_sequence
        self.input_type = input_type
        self.do_sample = do_sample
    
    
    ### Public Methods ###
    
    def autoregressive_generation(
        self, 
        x: Tensor, 
        num_steps: int, 
        concat_axis: int = 1, 
        stream: bool = False,
        preprocess_fn: Optional[Callable[[Tensor], Tensor]] = None,
        postprocess_fn: Optional[Callable[[Tensor], Tensor]] = None
    ) -> Union[Tensor, Generator[Tensor, None, None]]:
        """
        Autoregressive generation function to generate data.
        
        Parameters:
        - x (Tensor): The input tensor.
        - num_steps (int): The number of steps to generate.
        - concat_axis (int): The axis to concatenate the generated data.
        - stream (bool): Whether to generate the data in a streaming fashion.
        - preprocess_fn (Callable, optional): Function to normalize input before model forward pass.
        - postprocess_fn (Callable, optional): Function to denormalize model output.
        """
        
        # Disable gradient computation
        with no_grad():
            # Set the model to evaluation mode
            self.eval()
            
            # Stream requested
            if stream:
                # Return the generator to stream the data
                return self._autoregressive_step_loop(
                    x = x, 
                    num_steps = num_steps, 
                    concat_axis = concat_axis, 
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
                        preprocess_fn = preprocess_fn, 
                        postprocess_fn = postprocess_fn
                    )
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
        preprocess_fn: Optional[Callable[[Tensor], Tensor]] = None,
        postprocess_fn: Optional[Callable[[Tensor], Tensor]] = None
    ) -> Generator[Tensor, None, None]:
        """
        Autoregressive step loop to generate data.
        
        Parameters:
        - x (Tensor): The input tensor.
        - num_steps (int): The number of steps to generate.
        - concat_axis (int): The axis to concatenate the generated data.
        - preprocess_fn (Callable, optional): Function to normalize input before model forward pass.
        - postprocess_fn (Callable, optional): Function to denormalize model output.
        
        Yields:
        - Tensor: The generated data at each step.
        """
        
        # Iterate over the maximum number of new steps to generate
        for _ in range(num_steps):
            # Crop the input sequence to the sequence length if larger
            cropped_x = x[:, -self.sequence_length:, ...]
            
            # Apply normalization if provided
            if preprocess_fn is not None:
                cropped_x = preprocess_fn(cropped_x)
            
            # Get the prediction logits from the model
            out = self(cropped_x)
            
            # Apply denormalization if provided
            if postprocess_fn is not None:
                out = postprocess_fn(out)
                
            # Unsqueeze the output to add the time dimension if needed
            if out.data.ndim <= 2:
                # Add a new axis for the time dimension
                out = out.unsqueeze(axis=1)
            
            # If the model is not set to return the full sequence, take only the last output
            if not self.return_sequence:
                out = out[:, -1:, ...]
            
            # If the input type is discrete, apply softmax and sample or take argmax
            if self.input_type == "discrete":
                # Apply the softmax function to get the probabilities
                out = out.softmax(axis=-1)
                
                # If sampling is enabled, sample from the distribution or take the argmax
                if self.do_sample:
                    # Sample the next item from the distribution
                    out = Tensor(
                        np.array([
                            np.random.choice(out.shape()[-1], p=out.data[i, -1])
                            for i in range(out.shape()[0])
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