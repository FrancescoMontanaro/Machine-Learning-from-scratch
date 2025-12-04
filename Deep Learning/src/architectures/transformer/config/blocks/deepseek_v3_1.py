from typing import Literal
from dataclasses import dataclass

from .common import BaseAttentionConfig, MLPConfig


@dataclass(kw_only=True)
class LatentAttentionConfig(BaseAttentionConfig):
    """
    Configuration for the latent attention mechanism.
    
    Parameters:
    - q_lora_rank (int): Rank for the LoRA applied to the query projection.
    - qk_lora_rank (int): Rank for the LoRA applied to the combined query-key projection.
    - qk_nope_head_dim (int): Head dimension for the NOPE mechanism in the query-key projection.
    - qk_rope_head_dim (int): Head dimension for the RoPE mechanism in the query-key projection.
    - kv_lora_rank (int): Rank for the LoRA applied to the key-value projection.
    - v_head_dim (int): Head dimension for the value projection.
    - softmax_scale (float): Scaling factor applied before the softmax in attention computation.
    """
    
    q_lora_rank: int
    qk_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    kv_lora_rank: int
    v_head_dim: int
    softmax_scale: float


@dataclass
class MOEConfig:
    """
    Configuration for the Mixture of Experts (MoE) feed-forward network.
    
    Parameters:
    - top_k (int): Number of top experts to select for each input token.
    - n_groups (int): Number of groups to split the input tokens into for routing.
    - top_k_groups (int): Number of top groups to select for each input token.
    - score_function (Literal["softmax", "sigmoid"]): Scoring function to use for routing.
    - route_scale (float): Scaling factor for the routing scores.
    
    - n_routed_experts (int): Number of experts routed expert (in total).
    - n_shared_experts (int): Number of experts shared expert (in total).
    
    - mlp_config (MLPConfig): Configuration for the MLP used within each expert.
    """
    
    ### Gate parameters ###
    top_k: int
    n_groups: int
    top_k_groups: int
    score_function: Literal["softmax", "sigmoid"]
    route_scale: float
    
    ### MoE parameters ###
    n_routed_experts: int
    n_shared_experts: int
    
    ### MLP parameters ###
    mlp_config: MLPConfig


@dataclass
class DeepSeekTransformerBlockConfig:
    """
    Configuration for a DeepSeek Transformer block.
    
    Parameters:
    - attention_config (LatentAttentionConfig): Configuration for the latent attention mechanism.
    - ffn_config (MOEConfig): Configuration for the Mixture of Experts feed-forward network.
    """
    
    # Configuration parameters
    attention_config: LatentAttentionConfig
    ffn_config: MOEConfig