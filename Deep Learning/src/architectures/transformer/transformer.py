import gc
import numpy as np
from typing import Union, Generator

from .decoder import Decoder
from ...optimizers import Adam
from ...core import Tensor, Module
from .data_loader import DataLoader
from ...loss_functions import CrossEntropy
from ...core.utils.data_processing import concat
from ...core.utils.context_manager import no_grad


class Transformer(Module):
    
    ### Magic methods ###
    
    def __init__(self, vocab_size: int, n_embed: int, n_attention_heads: int, sequence_length: int, n_decoder_blocks: int = 4, dropout: float = 0.1, *args, **kwargs) -> None:
        """
        Initialize the transformer model.
        
        Parameters:
        - vocab_size (int): The size of the vocabulary.
        - n_embed (int): The embedding size.
        - n_attention_heads (int): The number of attention heads.
        - sequence_length (int): The sequence length of the input data.
        - n_decoder_blocks (int): The number of transformer blocks.
        - dropout (float): The dropout rate.
        """
        
        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        # Store the parameters
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        
        # Create the decoder module
        self.decoder: Decoder = Decoder( # (B, S) -> (B, S, V)
            vocab_size  = vocab_size,
            n_embed = n_embed,
            n_attention_heads = n_attention_heads,
            sequence_length = sequence_length,
            n_decoder_blocks = n_decoder_blocks,
            dropout = dropout
        )
    
    
    ### Public methods ###
    
    def fit(self, data_loader: DataLoader, epochs: int, lr: float, batch_size: int, eval_iters: int = 200) -> None:
        """
        Method to train the model.
        
        Parameters:
        - data_loader (DataLoader): The data loader object to get the data from.
        - epochs (int): The number of epochs.
        - lr (float): The learning rate.
        - batch_size (int): The batch size.
        - eval_iters (int): The number of iterations to evaluate the loss on the training and validation sets.
        """
        
        # Define the optimizer and loss function
        optimizer = Adam(learning_rate=lr, parameters=self.parameters(), weight_decay=0.01)
        loss_fn = CrossEntropy()
        
        # Iterate over the epochs
        for epoch in range(epochs):
            # Call the garbage collector to free up memory
            gc.collect()
            
            # Get the batch
            x, y = data_loader.get_batch(split='train', batch_size=batch_size, sequence_length=self.sequence_length) # (B, S), (B, S)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Get the logits and loss
            out = self(x) # (B, S, V)
            
            # Reshape the logits and labels to 2D
            out = out.reshape((-1, self.vocab_size)) # (B*S, V)
            y = y.reshape((-1,)) # (B*S,)
            
            # Compute the loss
            loss = loss_fn(y, out)
            
            # Perform the backward pass
            loss.backward()
            
            # Update the parameters
            optimizer.update()
            
            # Clear the computational graph
            loss.clear_graph()
            
            # Print the losses
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {loss.data:.4f}')
            
        
    def generate(self, input_tokens: Tensor, max_new_tokens: int, stream: bool = False) -> Union[Tensor, Generator[Tensor, None, None]]:
        """
        Method to generate new tokens (inference).
        
        Parameters:
        - input_tokens (Tensor): The input tokens.
        - max_new_tokens (int): The maximum number of new tokens to generate.
        - stream (bool): Whether to generate the tokens in a streaming fashion.
        
        Returns:
        - Union[Tensor, Generator[Tensor]]: The generated tokens or a generator to iterate over the generated tokens in a streaming fashion.
        """
        
        
        # Disable gradient computation
        with no_grad():
            # Set the model to evaluation mode
            self.eval()
            
            # Stream the generated tokens using a generator
            if stream:
                # Define the generator to stream the tokens
                def stream_tokens() -> Generator[Tensor, None, None]:
                    """
                    Generator to stream the generated tokens.
                    
                    Yields:
                    - Tensor: The next token generated.
                    """
                    
                    # Initialize the local tokens
                    local_tokens = input_tokens
                    
                    # Iterate over the maximum number of new tokens
                    for _ in range(max_new_tokens):
                        # Crop the input tokens to the sequence length if larger
                        cropped_input_tokens = local_tokens[:, -self.sequence_length:]
                        
                        # Get the predictions
                        logits = self(cropped_input_tokens)
                        
                        # Focus only on the last time step
                        logits = logits[:, -1, :]
                        
                        # Apply the softmax function to get the probabilities
                        probs = logits.softmax(axis=-1)
                        
                        # Sample the next token from the distribution
                        next_token = Tensor(
                            np.array([
                                np.random.choice(probs.shape()[1], p=probs.data[i])
                                for i in range(probs.shape()[0])
                            ]).reshape(-1, 1),
                            dtype=np.int32
                        )
                        
                        # Yield the next token
                        yield next_token
                        
                        # Concatenate the token for the next iteration
                        local_tokens = concat([local_tokens, next_token], axis=-1)
                
                # Return the generator to stream the tokens         
                return stream_tokens()
            
            # Generate all the tokens at once
            else:
                # Iterate over the maximum number of new tokens
                for _ in range(max_new_tokens):
                    # Crop the input tokens to the sequence length if it is larger
                    cropped_input_tokens = input_tokens[:, -self.sequence_length:] # (B, S)
                    
                    # Get the predictions
                    logits = self.forward(cropped_input_tokens)
                    
                    # Focus only on the last time step
                    logits = logits[:, -1, :]
                    
                    # Apply the softmax function to get the probabilities
                    probs = logits.softmax(axis=-1)
                    
                    # Sample the next token from the distribution
                    next_token = Tensor( # (B, 1)
                        np.array([
                            np.random.choice(probs.shape()[1], p=probs.data[i]) 
                            for i in range(probs.shape()[0])
                        ]).reshape(-1, 1),
                        dtype=np.int32
                    )
                        
                    # Concatenate the token
                    input_tokens = concat([input_tokens, next_token], axis=-1) # (B, S+1)
                    
                # Return the generated tokens
                return input_tokens


    ### Protected methods ###
            
    def _forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the transformer model.
        
        Parameters:
        - x (Tensor): The input tensor.
        
        Returns:
        - Tensor: The output tensor.
        """
        
        # Dimensions are:
        # - B: batch size
        # - S: sequence length
        # - V: vocabulary size
            
        # Run the forward pass of the decoder
        logits = self.decoder(x) # (B, S) -> (B, S, V)
        
        # Return the logits
        return logits # (B, S, V)
    
    
    def _lazy_init(self, x: Tensor) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - x (Tensor): Features of the dataset
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape()) == 2, f"Invalid input shape. Input must be a 2D array. The shape must be (Batch size, sequence length). Got shape: {x.shape()}"