from typing import Generator, Union

from ...core import Tensor
from ..base import Architecture
from ..sequential import Sequential
from ...core.utils.data_processing import concat
from ...core.utils.context_manager import no_grad


class AutoRegressive(Architecture):
    
    def __init__(self, sequence_length: int, *args, **kwargs) -> None:
        """
        Initialize the autoregressive architecture.
        
        Parameters:
        - sequence_length (int): The length of the input sequence.
        """
            
        # Initialize the autoregressive architecture with a sequence length.
        super().__init__(*args, **kwargs)
        
        # Set the sequence length
        self.sequence_length = sequence_length
    
    
    ### Public Methods ###
    
    def autoregressive_generation(self, x: Tensor, num_steps: int, concat_axis: int = 1, stream: bool = False) -> Union[Tensor, Generator[Tensor, None, None]]:
        """
        Autoregressive generation function to generate data.
        
        Parameters:
        - x (Tensor): The input tensor.
        - num_steps (int): The number of steps to generate.
        - concat_axis (int): The axis to concatenate the generated data.
        - stream (bool): Whether to generate the data in a streaming fashion.
        """
        
        # Disable gradient computation
        with no_grad():
            # Set the model to evaluation mode
            self.eval()
            
            # Stream requested
            if stream:
                # Return the generator to stream the data
                return self._autoregressive_step_loop(x, num_steps, concat_axis)
            
            # Generate all the data at once
            else:   
                # Return the generated data
                return self.concat_generation(self._autoregressive_step_loop(x, num_steps, concat_axis))
            
            
    @staticmethod
    def concat_generation(generator: Generator[Tensor, None, None], concat_axis: int = 1) -> Tensor:
        """
        Concatenate generation function to generate data.
        
        Parameters:
        - x (Tensor): The input tensor.
        - num_steps (int): The number of steps to generate.
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
            
    def _autoregressive_step_loop(self, x: Tensor, num_steps: int, concat_axis: int = 1, *args, **kwargs) -> Generator[Tensor, None, None]:
        """
        Autoregressive step loop to generate data.
        
        Parameters:
        - x (Tensor): The input tensor.
        - num_steps (int): The number of steps to generate.
        - concat_axis (int): The axis to concatenate the generated data.
        
        Yields:
        - Tensor: The generated data at each step.
        """
        
        # Iterate over the maximum number of new tokens
        for _ in range(num_steps):
            # Crop the input tokens to the sequence length if larger
            cropped_x = x[:, -self.sequence_length:, ...]
            
            # Get the predictions
            logits = self(cropped_x)
            
            # Add the time axis to the logits
            logits = logits.unsqueeze(1)
            
            # Yield the next token
            yield logits
            
            # Concatenate the token for the next iteration
            x = concat([x, logits], axis=concat_axis)


class SequentialAutoRegressive(AutoRegressive, Sequential):
    """
    Autoregressive architecture with Sequential architecture functionalities
    """
    
    pass