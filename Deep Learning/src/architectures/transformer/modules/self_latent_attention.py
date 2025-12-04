import numpy as np

from ....core import Tensor, Module
from ..config import LatentAttentionConfig
from ....layers import Dense, RMSNorm, Dropout, RoPE
from ....core.utils.data_processing import split, einsum


class SelfMultiHeadLatentAttention(Module):
    
    ### Magic methods ###
    
    def __init__(
        self,
        config: LatentAttentionConfig,
        *args, **kwargs
    ) -> None:
        """
        Class constructor for Self-MultiLatentAttention layer.
        
        Parameters:
        - config (LatentAttentionConfig): Configuration for the latent attention mechanism.
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Check that max_seq_len is specified
        assert config.max_seq_len is not None, "max_seq_len must be specified in the configuration."
        
        # Store the parameters
        self.num_heads = config.num_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_lora_rank = config.qk_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.softmax_scale = config.softmax_scale
        self.causal = config.causal
        self.dropout_rate = config.dropout
        self.max_seq_len = config.max_seq_len

        # Compute the dimension of the query/key head
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        
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
        
        # Define the RoPE layer for positional embeddings
        self.rope = RoPE(max_seq_len=self.max_seq_len)
        
        # Initialize the dropout layers
        self.attn_dropout = Dropout(self.dropout_rate)  # (B, S, num_heads, S) -> (B, S, num_heads, S)
        self.proj_dropout = Dropout(self.dropout_rate)  # (B, S, E) -> (B, S, E)

        # Initialize the output projection layer (will be initialized in lazy init)
        self.wo: Dense # (B, S, num_heads * Hv) -> (B, S, E)
        
        # Registering the attention mask as a buffer
        self.attention_mask: Tensor # (S, S) -> (S, S)
        
        # Caches for key, value and positional embeddings for faster inference
        self.kv_cache: Tensor # (B, S, num_heads, Hqk_nope + Hv)
        self.pe_cache: Tensor # (B, S, num_heads, Hqk_rope)
        
    
    ### Public methods ###
    
    def reset_cache(self) -> None:
        """
        Reset KV cache for new sequence generation.
        """
        
        # Fill the caches with zeros
        self.kv_cache.data.fill(0)
        self.pe_cache.data.fill(0)
        
    
    ### Protected methods ###

    def _forward(self, x: Tensor, start_pos: int = 0,  *args, **kwargs) -> Tensor:
        """
        Foward pass of the layer
        
        Parameters:
        - x (Tensor): Input tensor
        - start_pos (int): Starting position for the attention (used for caching during inference)

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
        q_nope, q_rope = split(q, [self.qk_nope_head_dim], axis=-1)
        q_rope = self.rope(q_rope, start_pos=start_pos) # (B, S, num_heads, Hqk_rope)
        
        ### Continue with the key and value projections ###
        
        # Compute the key and value projections
        kv = self.wkv_a(x) # (B, S, E) -> (B, S, r + Hqk_rope)
        kv, k_pe = split(kv, [self.kv_lora_rank], axis=-1) # (B, S, r), (B, S, Hqk_rope)
        k_pe: Tensor = self.rope(k_pe.unsqueeze(2), start_pos=start_pos) # (B, S, Hqk_rope)

        # Normalize the key-value representations
        kv_normalized = self.kv_norm(kv) # (B, S, r)

        # Compute the key and value projections using the low-rank representation
        wkv_b = self.wkv_b.weights
        wkv_b = wkv_b.reshape((self.num_heads, -1, self.kv_lora_rank))
        
        # Compute the normalized key-value representations
        if self.training:
            # In training mode, use the full sequence
            q_nope_proj = einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])

            # Compute the attention scores using the full sequence
            scores = (
                einsum("bshc,btc->bsht", q_nope_proj, kv_normalized) +
                einsum("bshr,bthr->bsht", q_rope, k_pe.expand(-1, -1, self.num_heads, -1))
            ) * self.softmax_scale
            
        else:
            # Use caching for faster inference
            end_pos = start_pos + S

            # Project the query using the low-rank representation
            q_nope_proj = einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            
            # Update the cache
            self.kv_cache[:B, start_pos:end_pos] = kv_normalized
            self.pe_cache[:B, start_pos:end_pos] = k_pe.squeeze(2)

            # Compute scores using the cache
            scores = (
                einsum("bshc,btc->bsht", q_nope_proj, self.kv_cache[:B, :end_pos]) +
                einsum("bshr,btr->bsht", q_rope, self.pe_cache[:B, :end_pos])
            ) * self.softmax_scale
        
        # If causal attention is enabled, we need to apply the attention mask
        if self.causal:
            if self.training:
                # Durante il training, usa la maschera completa
                scores = scores.masked_fill(self.attention_mask.data[:S, :S] == 0, float('-inf'))
            else:
                # Durante l'inferenza, la maschera deve considerare start_pos
                scores = scores.masked_fill(self.attention_mask.data[:S, :end_pos] == 0, float('-inf'))
            
        # Normalize the attention scores using softmax to get the attention weights
        attention_scores = scores.softmax(axis=-1) # (B, S, num_heads, S)
        
        # Apply dropout to the attention weights
        attention_scores = self.attn_dropout(attention_scores) # (B, S, num_heads, S)
        
        # Apply dropout to the attention weights
        if self.training:
            # Compute the output using the full sequence
            out = einsum("bsht,btc->bshc", attention_scores, kv_normalized)
        else:
            # Compute the output using the cache
            out = einsum("bsht,btc->bshc", attention_scores, self.kv_cache[:B, :end_pos])
        
        # Project the output to get the value representations
        out = einsum("bshc,hdc->bshd", out, wkv_b[:, -self.v_head_dim:]) # (B, S, num_heads, Hqk_nope + Hv) -> (B, S, num_heads, Hv)
        
        # Final linear projection
        out = self.wo(out.flatten(2)) # (B, S, num_heads * Hv) -> (B, S, E)
        
        # Apply dropout to the output projection
        out = self.proj_dropout(out) # (B, S, E)
        
        # Return the output of the layer
        return out # (B, S, E)
    
    
    def _lazy_init(self, x: Tensor, *args, **kwargs) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, sequence length, embedding size)
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape) == 3, f"Invalid input shape. Input must be a 3D array. The shape must be (Batch size, sequence length, embedding size). Got shape: {x.shape}"
        
        # Unpack the shape of the input data
        B, S, E = x.shape # (B, S, E)
        
        # Initialize the output projection layer
        self.wo = Dense(E, add_bias=False)
        
        # Initialize the wkv_b layer weights by calling it with a dummy input
        # This is necessary because we access wkv_b.weights directly in forward
        dummy_kv = Tensor(np.zeros((1, 1, self.kv_lora_rank)), requires_grad=False)
        _ = self.wkv_b(dummy_kv)
        
        # Initialize the key, value and positional embeddings caches
        self.register_buffer("kv_cache", Tensor(np.zeros((B, S, self.kv_lora_rank)))) # (B, S, P)
        self.register_buffer("pe_cache", Tensor(np.zeros((B, S, self.qk_rope_head_dim)))) # (B, S, P)
        
        # If causal attention is enabled, we need to create the attention mask
        if self.causal:
            # Initialize the attention mask as a lower triangular matrix for causal attention
            self.register_buffer("attention_mask", Tensor(np.tril(np.ones((S, S))))) # (S, S) -> (S, S)