import numpy as np

from ..core import Tensor, Module


class PositionalEncoding(Module):
    
    ### Magic methods ###
    
    def __init__(self, max_len: int, *args, **kwargs) -> None:
        """
        Class constructor
        
        Parameters:
        - max_len (int): Maximum sequence length
        - dropout (float): Dropout rate for regularization. Default is 0.1
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Store parameters
        self.max_len = max_len
        
        # Initialize the positional encoding matrix
        self.pos_encoding: Tensor
    
    
    ### Protected methods ###

    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass of the positional encoding layer
        
        Parameters:
        - x (Tensor): Input tensor with shape (batch_size, sequence_length, features)
        
        Returns:
        - Tensor: Input tensor with positional encoding added
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Store the input shape
        input_shape = x.shape()
        
        # Check if the input is 3D (batch_size, sequence_length, features)
        assert len(input_shape) == 3, f"Invalid input shape. Expected 3D tensor (batch_size, seq_len, features). Got shape: {input_shape}"
        
        # Unpack the input shape
        _, seq_len, _ = input_shape
        
        # Check if sequence length exceeds maximum
        assert seq_len <= self.max_len, f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
        
        # Extract the relevant portion of positional encoding
        pos_enc_slice = self.pos_encoding[:seq_len, :]
        
        # Add positional encoding to input (broadcasting over batch dimension)
        output = x + pos_enc_slice
        
        return output
    
        
    def _lazy_init(self, x: Tensor, *args, **kwargs) -> None:
        """
        Method to initialize the positional encoding matrix
        
        Parameters:
        - x (Tensor): Input tensor with shape (batch_size, sequence_length, features)
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape()) == 3, f"Invalid input shape. Expected 3D tensor (batch_size, seq_len, features). Got shape: {x.shape()}"
        
        # Extract dimensions
        _, _, d_model = x.shape()
        
        # Create positional encoding matrix
        pos_encoding = np.zeros((self.max_len, d_model))
        
        # Create position indices
        position = np.arange(0, self.max_len).reshape(-1, 1)
        
        # Create dimension indices for the sinusoidal pattern
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        
        # Apply cosine to odd indices
        if d_model % 2 == 1:
            pos_encoding[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        # Store as non-trainable parameter
        self.pos_encoding = Tensor(
            data = pos_encoding,
            requires_grad = False,
            is_parameter = False
        )