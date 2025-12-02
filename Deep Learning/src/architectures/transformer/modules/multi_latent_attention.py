import numpy as np

from ....core import Tensor, Module
from ....layers import Dense, RMSNorm
from ....core.utils.data_processing import split, einsum, apply_rope_embeddings


class SelfMultiHeadLatentAttention(Module):
    
    ### Magic methods ###
    
    def __init__(
        self,
        num_heads: int,
        q_lora_rank: int,
        qk_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        v_head_dim: int,
        softmax_scale: float,
        causal_attention: bool = False, 
        *args, **kwargs
    ) -> None:
        """
        Class constructor for Self-MultiLatentAttention layer.
        
        Parameters:
        - num_heads (int): Number of attention heads.
        - q_lora_rank (int): Rank for low-rank query projection.
        - qk_lora_rank (int): Rank for low-rank key/value projection.
        - qk_nope_head_dim (int): Head dimension for NoPE key/value projection (Without Positional Encoding).
        - qk_rope_head_dim (int): Head dimension for RoPE key/value projection.
        - kv_lora_rank (int): Rank for low-rank key/value projection.
        - v_head_dim (int): Head dimension for value projection.
        - softmax_scale (float): Scaling factor for softmax normalization.
        - causal_attention (bool): whether to use causal attention (default: False)
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Store the parameters
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.qk_lora_rank = qk_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.softmax_scale = softmax_scale
        self.causal_attention = causal_attention
        
        # Compute the dimension of the query/key head
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        
        # If q_lora_rank is 0, we use a standard Dense layer for the query projection
        if self.q_lora_rank == 0:
            # Define a single Dense layer for the query projection
            self.wq = Dense(self.num_heads * self.qk_head_dim, add_bias=False) # (B, S, E) -> (B, S, num_heads * Hqk)
            
        # Otherwise, we use a low-rank approximation for the query projection
        else:
            # Define two Dense layers for the low-rank query projection
            self.wq_a = Dense(self.q_lora_rank, add_bias=False) # (B, S, E) -> (B, S, r)
            self.q_norm = RMSNorm() # (B, S, E) -> (B, S, E)
            self.wq_b = Dense(self.num_heads * self.qk_head_dim, add_bias=False) # (B, S, r) -> (B, S, num_heads * Hqk)

        # Define a single Dense layer for the key projection in order to make it more computationally efficient
        self.wkv_a = Dense(self.kv_lora_rank + self.qk_rope_head_dim, add_bias=False) # (B, S, E) -> (B, S, r + Hqk_rope)
        self.kv_norm = RMSNorm() # (B, S, r) -> (B, S, r)
        self.wkv_b = Dense(self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), add_bias=False) # (B, S, r + Hqk_rope) -> (B, S, num_heads * (Hqk_nope + Hv))
        self.softmax_scale = self.qk_head_dim ** -0.5 # Scaling factor for softmax normalization
        
        # Initialize the output projection layer (will be initialized in lazy init)
        self.wo: Dense # (B, S, num_heads * Hv) -> (B, S, E)
        
        # Registering the attention mask as a buffer
        self.attention_mask: Tensor # (S, S) -> (S, S)
        
        # Caches for key, value and positional embeddings for faster inference
        self.kv_cache: Tensor # (B, S, num_heads, Hqk_nope + Hv)
        self.pe_cache: Tensor # (B, S, num_heads, Hqk_rope)
        
    
    ### Protected methods ###

    def _forward(self, x: Tensor, start_pos: int, freq_cis: Tensor, *args, **kwargs) -> Tensor:
        """
        Foward pass of the layer
        
        Parameters:
        - x (Tensor): Input tensor
        - start_pos (int): Starting position for the attention (used for caching during inference)
        - freq_cis (Tensor): Precomputed frequency embeddings for RoPE (if any)

        Returns:
        - Tensor: Output of the layer
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Dimensions are:
        # - B: batch size
        # - S: sequence length
        # - E: embedding size (embedding dimension of the original data)
        
        # Unpack the shape of the input data
        B, S, _ = x.shape # (B, S, E)
        
        # Compute the end position for the attention
        end_pos = start_pos + S
        
        ### Start with the query projection ###
        
        # Compute the query projection
        if self.q_lora_rank == 0:
            # Standard query projection
            q = self.wq(x) # (B, S, E) -> (B, S, num_heads * Hqk)
        else:
            # Low-rank query projection
            q = self.q_norm(self.wq_a(x)) # (B, S, E) -> (B, S, r)
            q = self.wq_b(q) # (B, S, r) -> (B, S, num_heads * (Hqk_nope + Hv))

        # Reshape the query tensor to separate the heads
        # and separate the NoPE and RoPE parts
        q = q.reshape((B, S, self.num_heads, self.qk_head_dim)) # (B, S, num_heads, Hqk)
        q_nope, q_rope = split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1)
        q_rope = apply_rope_embeddings(q_rope, freq_cis) # (B, S, num_heads, Hqk_rope)
        
        ### Continue with the key and value projections ###
        
        # Compute the key and value projections
        kv = self.wkv_a(x) # (B, S, E) -> (B, S, r + Hqk_rope)
        kv, k_pe = split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], axis=-1) # (B, S, r), (B, S, Hqk_rope)
        k_pe: Tensor = apply_rope_embeddings(k_pe.unsqueeze(2), freq_cis) # (B, S, Hqk_rope)

        # Compute the key and value projections using the low-rank representation
        wkv_b = self.wkv_b.weights
        wkv_b = wkv_b.reshape((self.num_heads, -1, self.kv_lora_rank))
        q_nope = einsum("bshd,bthd->bsht", q_nope, wkv_b[:, :self.qk_nope_head_dim]) # (B, S, num_heads, Hqk_nope)
        
        # Update the key and value caches
        self.kv_cache[:B, start_pos:end_pos] = self.kv_norm(kv)
        self.pe_cache[:B, start_pos:end_pos] = k_pe.squeeze(2)
        
        # Compute the attention scores
        scores = ( # (B, S, num_heads, Hqk_nope) x (B, S, num_heads, Hqk_nope) -> (B, S, num_heads, S)
            einsum("bshc,btc->bsht", q_nope, self.kv_cache[:B, :end_pos]) +
            einsum("bshr,btr->bsht", q_rope, self.pe_cache[:B, :end_pos])
        ) * self.softmax_scale
        
        # If causal attention is enabled, we need to apply the attention mask
        if self.causal_attention:
            # Apply the attention mask to the scores
            scores = scores.masked_fill(self.attention_mask.data[:S, :S] == 0, float('-inf'))
            
        # Normalize the attention scores using softmax to get the attention weights
        attention_scores = scores.softmax(axis=-1) # (B, S, num_heads, S)
        
        # Apply dropout to the attention weights
        out = einsum("bsht,btc->bshc", attention_scores, self.kv_cache[:B, :end_pos]) # (B, S, num_heads, Hqk_nope + Hv)
        out = einsum("bshc,hdc->bshd", out, wkv_b[:, -self.v_head_dim:]) # (B, S, num_heads, Hqk_nope + Hv) -> (B, S, num_heads, Hv)
        
        # Final linear projection
        return self.wo(out.flatten(2)) # (B, S, num_heads * Hv) -> (B, S, E)
    
    
    def _lazy_init(self, x: Tensor) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, sequence length, embedding size)
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape) == 3, f"Invalid input shape. Input must be a 3D array. The shape must be (Batch size, sequence length, embedding size). Got shape: {x.shape}"
        
        # If causal attention is enabled, we need to create the attention mask
        if self.causal_attention:
            # Unpack the shape of the input data
            B, S, E = x.shape # (B, S, E)
            
            # Initialize the output projection layer
            self.wo = Dense(E, add_bias=False)
            
            # Initialize the attention mask as a lower triangular matrix for causal attention
            self.register_buffer("attention_mask", Tensor(np.tril(np.ones((S, S))))) # (S, S) -> (S, S)
            
            # Initialize the key, value and positional embeddings caches
            self.register_buffer("kv_cache", Tensor(np.zeros((B, S, self.kv_lora_rank)))) # (B, S, P)
            self.register_buffer("pe_cache", Tensor(np.zeros((B, S, self.qk_rope_head_dim)))) # (B, S, P)