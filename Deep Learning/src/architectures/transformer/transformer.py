import time
import numpy as np
from typing import Generator

from ...core import Tensor
from .decoder import Decoder
from ...optimizers import Adam
from .data_loader import DataLoader
from ...loss_functions import CrossEntropy
from ..auto_regressive import AutoRegressive
from ...core.utils.data_processing import concat
from ...core.utils.context_manager import no_grad


class Transformer(AutoRegressive):
    
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
    
    
    def fit(
        self, 
        data_loader: DataLoader, 
        steps: int, lr: float, 
        batch_size: int, 
        eval_iters: int = 10,
        grad_accumulation_steps: int = 1
    ) -> dict[str, Tensor]:
        """
        Method to train the model.
        
        Parameters:
        - data_loader (DataLoader): The data loader object to get the data from.
        - steps (int): The number of training steps.
        - lr (float): The learning rate.
        - batch_size (int): The batch size.
        - eval_iters (int): The number of iterations to evaluate the loss on the training and validation sets. If it is less than the number of gradient accumulation steps, it will be set to the number of gradient accumulation steps.
        - grad_accumulation_steps (int): The number of gradient accumulation steps.
        """
        
        # Ensure that eval_iters is greater than or equal to grad_accumulation_steps
        eval_iters = max(eval_iters, grad_accumulation_steps)
        
        # Define the optimizer and loss function
        optimizer = Adam(learning_rate=lr, parameters=self.parameters(), weight_decay=0.01)
        loss_fn = CrossEntropy()
        
        # Initialize the history of the model
        self.init_history()
        
        # Initialize the control variables
        elapsed_time = 0.0
        
        # Iterate over the steps
        for step in range(steps):
            # Start the timer
            start_time = time.time()
            
            # Call the garbage collector to free up memory
            self.clear_cache()
            
            # If the step is a multiple of eval_iters, evaluate the loss on the validation set
            if (step + 1) % eval_iters == 0:
                # Count the number of tensors in memory
                tensors_in_memory = self.count_tensors_in_memory()
                
                # Estimate the losses
                self.evaluate_losses(data_loader, eval_iters, batch_size)
                
                # Compute the time statistics
                elapsed_time += time.time() - start_time
                ms_per_step = (elapsed_time / (step + 1)) * 1000
                
                # Print the validation loss
                print(f'Step {step+1}/{steps} | {tensors_in_memory} tensors in memory | {ms_per_step:.2f} ms/step - Train Loss: {self.history["loss"].data[-1]:.4f} | Validation loss: {self.history["val_loss"].data[-1]:.4f}')
            
            # Get the batch
            x, y = data_loader.get_batch(split='train', batch_size=batch_size, sequence_length=self.sequence_length) # (B, S), (B, S)
            
            # Get the logits and loss
            out = self(x) # (B, S, V)
            
            # Reshape the logits and labels to 2D
            out = out.reshape((-1, self.vocab_size)) # (B*S, V)
            y = y.reshape((-1,)) # (B*S,)
            
            # Compute the loss
            loss = loss_fn(y, out)
            
            # Check if the gradient accumulation steps are greater than 1
            if grad_accumulation_steps > 1:
                # Scale the loss for gradient accumulation
                loss /= grad_accumulation_steps
                
            # Backpropagate the loss
            loss.backward()
            
            # If the number of accumulation steps is reached or it is the last step, update the parameters
            if step % grad_accumulation_steps == 0 or step == steps - 1:
                # Update the parameters
                optimizer.update()
                
                # Zero the gradients
                optimizer.zero_grad()
            
        # Return the training history
        return self.history
    
    
    def evaluate_losses(self, data_loader: DataLoader, eval_iters: int, batch_size: int) -> None:
        """
        Method to evaluate the losses on the training and validation sets and store them in the history.
        
        Parameters:
        - data_loader (DataLoader): The data loader object to get the data from.
        - eval_iters (int): The number of iterations to evaluate the loss.
        - batch_size (int): The batch size.
        """
                
        # Initialize the loss function
        loss_fn = CrossEntropy()
        
        # Set the model to evaluation mode
        self.eval()
        
        # Disable gradient computation
        with no_grad():
            # Iterate over the splits
            for split in ['train', 'validation']:
                # Initialize the losses tensor
                losses = np.zeros(eval_iters)
                
                # Iterate over the evaluation iterations
                for iter in range(eval_iters):
                    # Call the garbage collector to free up memory
                    self.clear_cache()
                    
                    # Getting a batch of data
                    x, y = data_loader.get_batch(split=split, batch_size=batch_size, sequence_length=self.sequence_length) # type: ignore
                    
                    # Get the logits and loss
                    logits = self(x)
                    
                    # Reshape the logits and labels to 2D
                    logits = logits.reshape((-1, self.vocab_size)) # (B*S, V)
                    y = y.reshape((-1,)) # (B*S,)
                    
                    # Compute the loss
                    loss = loss_fn(y, logits)
                    
                    # Store the loss
                    losses[iter] = loss.detach().to_numpy()
                
                # Compute the mean loss and store it in the history
                history_loss_name = "loss" if split == "train" else "val_loss"
                self.history[history_loss_name].data = np.append(self.history[history_loss_name].data, losses.mean())
            
        # Set the model back to training mode
        self.train()
            

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
        
        
    def _autoregressive_step_loop(self, x: Tensor, num_steps: int, concat_axis: int = -1, *args, **kwargs) -> Generator[Tensor, None, None]:
        """
        Autoregressive step loop to generate data.
        
        Parameters:
        - x (Tensor): The input tensor.
        - num_steps (int): The number of steps to generate.
        - concat_axis (int): The axis to concatenate the generated data.
        
        Yields:
        - Tensor: The next token generated.
        """
        
        # Iterate over the maximum number of new tokens
        for _ in range(num_steps):
            # Crop the input tokens to the sequence length if larger
            cropped_input_tokens = x[:, -self.sequence_length:]
            
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
            x = concat([x, next_token], axis=concat_axis)