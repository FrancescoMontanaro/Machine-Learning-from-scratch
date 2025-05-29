import math
import time
import numpy as np
from typing import Generator, Union, Optional, Callable, Literal, Dict

from ...core import Tensor
from .decoder import Decoder
from .encoder import Encoder
from ...optimizers import Optimizer
from ...loss_functions import LossFn
from .encoder_decoder import EncoderDecoder
from ..auto_regressive import AutoRegressive
from ...core.utils.context_manager import no_grad
from ...core.utils.data_processing import concat, shuffle_data


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
            return_sequence = return_sequence
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
            return_sequence = return_sequence
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
            decoder_causal_attention = decoder_causal_attention
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
    
    def fit(
        self, 
        X_train: Tensor, 
        y_train: Tensor,
        optimizer: Optimizer,
        loss_fn: LossFn,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        epochs: int = 10,
        metrics: list[Callable[..., Tensor]] = [],
        callbacks: list[Callable] = [],
        shuffle_between_epochs: bool = True,
        X_valid: Optional[Tensor] = None,
        y_valid: Optional[Tensor] = None,
        *args,
        X_encoder_train: Optional[Tensor] = None,
        X_encoder_valid: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, list[Tensor]]:
        """
        Method to train the neural network
        
        Parameters:
        - X_train (Tensor): Features of the training dataset. Shape: (samples, ...)
        - y_train (Tensor): Labels of the training dataset. Shape: (samples, ...)
        - X_valid (Tensor): Features of the validation dataset. Shape: (samples, ...)
        - y_valid (Tensor): Labels of the validation dataset. Shape: (samples, ...)
        - optimizer (Optimizer): Optimizer to update the parameters of the model
        - loss_fn (LossFn): Loss function to compute the error of the model
        - batch_size (int): Number of samples to use for each batch. Default is 8
        - gradient_accumulation_steps (int): Number of steps to accumulate the gradients before updating the parameters. Default is 1
        - epochs (int): Number of epochs to train the model. Default is 10
        - metrics (list[Callable]): List of metrics to evaluate the model. Default is an empty list
        - callbacks (list[Callback]): List of callbacks to execute
        - shuffle_between_epochs (bool): Flag to shuffle the data between epochs. Default is True
        - X_encoder_train (Tensor): Features of the encoder training dataset. Shape: (samples, ...). If None, the model will use the same features as the decoder training dataset
        - X_encoder_valid (Optional[Tensor]): Features of the encoder validation dataset. Shape: (samples, ...). Default is None
        
        Returns:
        - Dict[str, list[Tensor]]: Dictionary containing the history of the model
        
        Raises:
        - ValueError: If the validation set is provided but the validation target is not provided
        """
        
        ############################
        ### Check the input data ###
        ############################
        
        # If the vaidation set is provided check if the validation target is provided
        if any([s is not None for s in [X_valid, X_encoder_valid]]) and not all([s is not None for s in [X_valid, X_encoder_valid, y_valid]]):
            raise ValueError("If a validation set is provided, the validation target must also be provided.")
        
        # If the encoder training data is not provided, use the decoder training data
        if X_encoder_train is None:
            X_encoder_train = X_train
        
        #######################
        ### Initializations ###
        #######################
        
        # Initialize the history of the model
        self.init_history(metrics)
        
        # Initialize the control variables
        self.epoch, self.stop_training = 0, False
        n_training_steps = max(1, math.ceil(X_train.shape()[0] / batch_size))
        n_valid_steps = max(1, math.ceil(X_valid.shape()[0] / batch_size)) if X_valid is not None else 0
        
        # Execute a first forward pass in evaluation mode to initialize the parameters and their shapes
        with no_grad():
            # Set the model in evaluation mode
            self.eval()
            
            # Compute the output of the model
            self(x=X_train[:1], encoder_input=X_encoder_train[:1], *args, **kwargs)
        
        # Set the parameters of the optimizer
        optimizer.set_parameters(self.parameters())
            
        ################################
        ### Start main training loop ###
        ################################
        
        # Iterate over the epochs
        while self.epoch < epochs and not self.stop_training:
            
            ############################
            ### Start training phase ###
            ############################
            
            # Set the model in training mode
            self.train()
            
            # Shuffle the dataset at the beginning of each epoch
            (X_train_shuffled, Y_train_shuffled), _ = shuffle_data((X_train, y_train)) if shuffle_between_epochs else (X_train, y_train)
            X_encoder_train_shuffled, _ = shuffle_data(X_encoder_train) if shuffle_between_epochs and X_encoder_train else X_encoder_train
            
            # Iterate over the batches
            elapsed_time = 0.0
            training_epoch_loss = Tensor(0.0, requires_grad=False)
            train_metrics = {metric.__name__: Tensor(0.0, requires_grad=False) for metric in metrics}
            for training_step in range(n_training_steps):
                # Store the start time
                start_time = time.time()
                
                # Get the current batch of data
                X_training_batch = X_train_shuffled[training_step * batch_size:(training_step + 1) * batch_size]
                X_encoder_training_batch = X_encoder_train_shuffled[training_step * batch_size:(training_step + 1) * batch_size]
                y_training_batch = Y_train_shuffled[training_step * batch_size:(training_step + 1) * batch_size]
                
                # Forward pass: Compute the output of the model
                training_batch_output = self.forward(x=X_training_batch, encoder_input=X_encoder_training_batch, *args, **kwargs)
                
                # Compute the loss of the model
                training_loss = loss_fn(y_training_batch, training_batch_output)
                    
                # Divide the loss by the number of gradient accumulation steps and execute the backward pass
                (training_loss / gradient_accumulation_steps).backward()
                
                # If the number of accumulation steps is reached or it is the last step, update the parameters
                if (training_step + 1) % gradient_accumulation_steps == 0 or training_step == n_training_steps - 1:
                    # Update the parameters of the model
                    optimizer.update()
                    
                    # Zero the gradients of the parameters
                    optimizer.zero_grad()
                    
                # Update the epoch loss
                training_epoch_loss += training_loss.detach()
                
                # Compute the metrics
                for metric in metrics:
                    train_metrics[metric.__name__] += metric(y_training_batch.detach(), training_batch_output.detach())
                        
                # Comute the the statistics
                end_time = time.time() # Store the end time
                elapsed_time += (end_time - start_time) # Update the elapsed time
                ms_per_step = elapsed_time / (training_step + 1) * 1000 # Compute the milliseconds per step
                tensors_in_memory = self.count_tensors_in_memory() # Compute the number of tensors in memory
                
                # Display epoch progress
                print(f"\rEpoch {self.epoch + 1}/{epochs} ({round((((training_step + 1)/n_training_steps)*100), 2)}%) | {tensors_in_memory} tensors in memory | {round(ms_per_step, 2)} ms/step --> loss: {training_loss.to_numpy():.5g}", end="")
            
            # Store the loss in the history
            self.history["loss"].append(training_epoch_loss / n_training_steps)
            
            # Compute the training metrics
            for metric in metrics:
                # Compute the average of the metrics for the training set and store them
                self.history[metric.__name__].append(train_metrics[metric.__name__] / n_training_steps)
            
            ##############################
            ### Start validation phase ###
            ##############################
            
            # Check if the validation set is provided
            if X_valid is not None and X_encoder_valid is not None and y_valid is not None:
                # Set the model in evaluation mode
                self.eval()
                
                # Disable automatic gradient computation
                with no_grad(): 
                    # Iterate over the validation steps
                    valid_epoch_loss = Tensor(0.0, requires_grad=False)
                    valid_metrics = {metric.__name__: Tensor(0.0, requires_grad=False) for metric in metrics}
                    for valid_step in range(n_valid_steps):
                        # Get the current batch of validation data
                        X_valid_batch = X_valid[valid_step * batch_size:(valid_step + 1) * batch_size]
                        y_valid_batch = y_valid[valid_step * batch_size:(valid_step + 1) * batch_size]
                        X_encoder_valid_batch = X_encoder_valid[valid_step * batch_size:(valid_step + 1) * batch_size]
                    
                        # Compute the output of the model for the current validation batch
                        valid_batch_output = self.forward(x=X_valid_batch, encoder_input=X_encoder_valid_batch, *args, **kwargs)
                        
                        # Compute the loss of the model for the current validation batch
                        # and update the validation epoch loss
                        valid_epoch_loss += loss_fn(y_valid_batch, valid_batch_output).detach()
                        
                        # Compute the metrics
                        for metric in metrics:
                            valid_metrics[metric.__name__] += metric(y_valid_batch, valid_batch_output).detach()
                
                # Store the validation losses in the history
                self.history["val_loss"].append(valid_epoch_loss / n_valid_steps)
            
                # Compute the average metrics for the validation set
                for metric in metrics:
                    # Compute the average of the metrics for the validation set and store them
                    self.history[f"val_{metric.__name__}"].append(valid_metrics[metric.__name__] / n_valid_steps)
        
            #############################
            ### Display the progress  ###
            #############################
            
            # Display progress with metrics
            print(
                f"\rEpoch {self.epoch + 1}/{epochs} --> "
                f"loss: {self.history['loss'][-1].to_numpy().item():.5g}"
                + "".join(
                    [f" - {metric.__name__.replace('_', ' ')}: {self.history[metric.__name__][-1].to_numpy().item():.5g}" for metric in metrics]
                ) +
                (
                    f" | Valid loss: {self.history['val_loss'][-1].to_numpy().item():.5g}"
                    + "".join(
                        [f" - Valid {metric.__name__.replace('_', ' ')}: {self.history[f'val_{metric.__name__}'][-1].to_numpy().item():.5g}" for metric in metrics]
                    ).ljust(50)   
                ) if X_valid is not None and y_valid is not None else "".ljust(50)
            )
            
            #############################
            ### Execute the callbacks ###
            #############################
            
            # Increment the epoch counter
            self.epoch += 1
                    
            # Execute the callbacks
            for callback in callbacks:
                # Call the callback
                callback(self)
                
            # Call the garbage collector to free up memory
            self.clear_cache()
         
        # Return the history of the training   
        return self.history
    
        
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
                        np.random.choice(out.shape()[-1], p=out.data[i, -1])
                        for i in range(out.shape()[0])
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