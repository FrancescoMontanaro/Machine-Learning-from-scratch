from typing import Optional, Union

from .mlp import MLP
from .moe import MoE
from ....core import Tensor, Module
from ....layers import LayerNormalization, RMSNorm
from .self_attention import SelfMultiHeadAttention
from .cross_attention import CrossMultiHeadAttention
from .self_latent_attention import SelfMultiHeadLatentAttention
from ..config import TransformerBlockConfig, DeepSeekTransformerBlockConfig, AttentionConfig


class Block(Module):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        config: Union[TransformerBlockConfig, DeepSeekTransformerBlockConfig],
        *args, **kwargs
    ) -> None:
        """
        Initialize the transformer's decoder block.
        
        Parameters:
        - config (Union[TransformerBlockConfig, DeepSeekTransformerBlockConfig]): The configuration for the transformer block.
        """
        
        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        # Handle a Standard Transformer Block
        if isinstance(config, TransformerBlockConfig):
            # Store the configuration parameters
            self.n_heads = config.attention_config.num_heads
            self.attention_config = config.attention_config

            # Define the self-attention mechanism
            self.layer_norm_1 = LayerNormalization() # (B, S, E) -> (B, S, E)
            self.self_attention_heads = SelfMultiHeadAttention(config=config.attention_config) # (B, S, E) -> (B, S, E)
            
            ### Start Optional cross-attention mechanism ###
            
            # Define the cross-attention mechanism
            self.layer_norm_cross: LayerNormalization # (B, S, E) -> (B, S, E)
            self.cross_attention_heads: CrossMultiHeadAttention # (B, S, E) -> (B, S, E)

            ### End Optional cross-attention mechanism ###
                
            # Define the MLP module
            self.ffn: MLP = MLP(config=config.ffn_config) # (B, S, E) -> (B, S, E)
            self.layer_norm_2 = LayerNormalization() # (B, S, E) -> (B, S, E)
            
            # Set the forward and lazy_init methods
            self._forward = self._forward_vanilla_transformer_block
            self._lazy_init = self._lazy_init_vanilla_transformer_block
            
        # Handle a DeepSeek Transformer Block
        elif isinstance(config, DeepSeekTransformerBlockConfig):
            # Initialize the self-latent attention mechanism
            self.mla = SelfMultiHeadLatentAttention(config=config.attention_config)
            
            # Initialize the MoE feed-forward network
            self.ffn_moe = MoE(config=config.ffn_config)
            
            # Initialize layer normalizations
            self.attn_norm = RMSNorm()
            self.ffn_norm = RMSNorm()
        
            # Set the forward and lazy_init methods
            self._forward = self._forward_deepseek_transformer_block
            self._lazy_init = self._lazy_init_deepseek_transformer_block
        
        else:
            # Handle other types of transformer blocks
            raise ValueError("Invalid config type. Must be either TransformerBlockConfig or DeepSeekTransformerBlockConfig.")
         
       
    ### Protected methods ###
    
    def _forward_vanilla_transformer_block(
        self, 
        x: Tensor,
        encoder_output: Optional[Tensor] = None,
        *args, **kwargs
    ) -> Tensor:
        """
        Forward pass of the vanilla transformer block.
        
        Parameters:
        - x (Tensor): The input x.
        - start_pos (int): Starting position (unused in vanilla, kept for API compatibility). Default is 0.
        - encoder_output (Optional[Tensor]): The output from the encoder (only for decoder blocks with cross-attention).
        
        Returns:
        - Tensor: The output embeddings.
        
        Raises:
        - ValueError: If cross-attention is used but encoder_output is not provided.
        """
        
        # Dimensions are:
        # - B: batch size
        # - S: sequence length
        # - E: embedding size (embedding dimension of the original data)
        
        # Apply the multi-attention mechanism with skip connections
        out = x + self.self_attention_heads(self.layer_norm_1(x)) # (B, S, E) + (B, S, E) -> (B, S, E)
        
        # Check if cross-attention has to be applied
        if isinstance(encoder_output, Tensor):
            # Apply cross-attention with residual connection
            out = out + self.cross_attention_heads(self.layer_norm_cross(out), encoder_output) # (B, S, E) + (B, S, E) -> (B, S, E)
        
        # Apply the ffn module with skip connections
        return out + self.ffn(self.layer_norm_2(out)) # (B, S, E) + (B, S, E) -> (B, S, E)
    
    
    def _forward_deepseek_transformer_block(
        self, 
        x: Tensor,
        start_pos: int = 0,
        *args, **kwargs
    ) -> Tensor:
        """
        Forward pass of the vanilla transformer block.
        
        Parameters:
        - x (Tensor): The input x.
        - start_pos (int): The starting position for the latent attention. Default is 0.
        - encoder_output (Optional[Tensor]): The output from the encoder (only for decoder blocks with cross-attention).
        
        Returns:
        - Tensor: The output embeddings.
        
        Raises:
        - ValueError: If cross-attention is used but encoder_output is not provided.
        """
        
        # Dimensions are:
        # - B: batch size
        # - S: sequence length
        # - E: embedding size (embedding dimension of the original data)
        
        # Apply the self multi-head latent attention with skip connections
        x = x + self.mla(self.attn_norm(x), start_pos) # (B, S, E) + (B, S, E) -> (B, S, E)
        
        # Apply the MoE feed-forward network with skip connections
        x = x + self.ffn_moe(self.ffn_norm(x)) # (B, S, E) + (B, S, E) -> (B, S, E)
        
        # Return the output
        return x
    
    
    def _lazy_init_vanilla_transformer_block(self, x: Tensor, encoder_output: Optional[Tensor] = None, *args, **kwargs) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, sequence length, embedding size)
        - encoder_output (Optional[Tensor]): Encoder output data. Shape: (Batch size, encoder sequence length, embedding size)
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape) == 3, f"Invalid input shape. Input must be a 3D array. The shape must be (Batch size, sequence length, embedding size). Got shape: {x.shape}"
        
        # Store the input shape of the layer
        _, _, E = x.shape # (B, S, E)
        
        # Check if the embedding size is divisible by the number of heads
        assert E % self.n_heads == 0, f"Embedding size {E} must be divisible by the number of heads {self.n_heads}."
        
        # Initialize the cross-attention mechanism if encoder_output is provided
        if isinstance(encoder_output, Tensor):
            # Create the cross-attention mechanism based on the attention_config type
            assert isinstance(self.attention_config, AttentionConfig), "attention_config must be an instance of AttentionConfig."
            
            # Initialize the cross-attention mechanism
            self.layer_norm_cross = LayerNormalization() # (B, S, E) -> (B, S, E)
            self.cross_attention_heads = CrossMultiHeadAttention(config=self.attention_config) # (B, S, E) -> (B, S, E)


    def _lazy_init_deepseek_transformer_block(self, x: Tensor, encoder_output: Optional[Tensor] = None, *args, **kwargs) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - x (Tensor): Input data. Shape: (Batch size, sequence length, embedding size)
        - encoder_output (Optional[Tensor]): Encoder output data. Shape: (Batch size, encoder sequence length, embedding size)
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape) == 3, f"Invalid input shape. Input must be a 3D array. The shape must be (Batch size, sequence length, embedding size). Got shape: {x.shape}"
        
        # Store the input shape of the layer
        _, _, E = x.shape # (B, S, E)
        
        # Initialize the cross-attention mechanism if encoder_output is provided
        if isinstance(encoder_output, Tensor):
            # Raise not implemented error
            raise NotImplementedError("Cross-attention in DeepSeek transformer block is not yet implemented.")