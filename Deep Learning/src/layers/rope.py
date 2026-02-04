import numpy as np

from ..core import Tensor, SingleOutputModule


class RoPE(SingleOutputModule):
    """
    Rotary Positional Embedding (RoPE) layer.    
    Applies rotational position encoding to input tensors, allowing the model
    to encode relative position information directly into the attention mechanism.
    """
    
    ### Magic methods ###
    
    def __init__(self, max_seq_len: int, theta: float = 10000.0, *args, **kwargs) -> None:
        """
        Initialize the Rotary Positional Embedding (RoPE) layer.
        
        Parameters:
        - dim (int): The dimension of the RoPE embeddings (must match the last dim of input).
        - max_seq_len (int): The maximum sequence length for which to precompute the embeddings.
        - theta (float): The base for the frequency computation (default 10000.0).
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Store parameters
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        # Buffers to hold precomputed frequencies
        self.freqs_cos: Tensor
        self.freqs_sin: Tensor
        
    
    ### Protected methods ###

    def _forward(self, x: Tensor, start_pos: int = 0, *args, **kwargs) -> Tensor:
        """
        Apply rotary positional embeddings to the input tensor.

        Parameters:
        - x (Tensor): Input tensor. Shape: (Batch, Seq, Heads, Dim) or (Batch, Seq, Dim)
        - start_pos (int): Starting position for slicing frequencies (for inference with KV cache)
        
        Returns:
        - Tensor: Tensor with rotary embeddings applied, same shape as input.
        """
        
        # Get sequence length from input
        if len(x.shape) == 4:
            # (Batch, Seq, Heads, Dim)
            _, S, _, D = x.shape
            has_heads = True
            
        elif len(x.shape) == 3:
            # (Batch, Seq, Dim)
            _, S, D = x.shape
            has_heads = False
            
        else:
            raise ValueError(f"Expected 3D or 4D input, got shape {x.shape}")
        
        # Get the relevant slice of precomputed frequencies
        end_pos = start_pos + S
        cos = self.freqs_cos.data[start_pos:end_pos] # (S, dim//2)
        sin = self.freqs_sin.data[start_pos:end_pos] # (S, dim//2)
        
        # Reshape for broadcasting
        if has_heads:
            # (S, dim//2) -> (1, S, 1, dim//2)
            cos = cos.reshape(1, S, 1, -1)
            sin = sin.reshape(1, S, 1, -1)
        else:
            # (S, dim//2) -> (1, S, dim//2)
            cos = cos.reshape(1, S, -1)
            sin = sin.reshape(1, S, -1)
        
        # Split input into two halves along the last dimension
        half_dim = D // 2
        x_real = x[..., :half_dim]  # First half
        x_imag = x[..., half_dim:]  # Second half
        
        # Apply rotation using complex multiplication:
        # (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos
        
        # Concatenate back together
        return Tensor.concat([out_real, out_imag], axis=-1)
    
    
    def _lazy_init(self, x: Tensor, *args, **kwargs) -> None:
        """
        Precompute the cos/sin frequencies for all positions.
        
        Parameters:
        - x (Tensor): Input tensor to infer dimension.
        """
        
        # Infer dimension from input
        dim = x.shape[-1]
        
        # Compute inverse frequencies for each dimension pair
        # freqs[i] = 1 / (theta^(2i/dim))
        inv_freq = 1.0 / (self.theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))

        # Create position indices [0, 1, 2, ..., max_seq_len-1]
        positions = np.arange(self.max_seq_len, dtype=np.float32)
        
        # Outer product: angles[pos, i] = pos * inv_freq[i]
        # Shape: (max_seq_len, dim // 2)
        angles = np.outer(positions, inv_freq)
        
        # Compute cos and sin
        # Shape: (max_seq_len, dim // 2) each
        freqs_cos = np.cos(angles)
        freqs_sin = np.sin(angles)
        
        # Register as buffers (not trainable parameters)
        self.register_buffer("freqs_cos", Tensor(freqs_cos, dtype=np.float32, requires_grad=False))
        self.register_buffer("freqs_sin", Tensor(freqs_sin, dtype=np.float32, requires_grad=False))