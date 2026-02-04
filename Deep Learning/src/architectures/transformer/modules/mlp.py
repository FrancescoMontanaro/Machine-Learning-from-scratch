from ..config import MLPConfig
from ....activations import ReLU
from ....layers import Dense, Dropout
from ....core import Tensor, SingleOutputModule


class MLP(SingleOutputModule):
    
    ### Magic methods ###
    
    def __init__(
        self,
        config: MLPConfig,
        *args, **kwargs
    ) -> None:
        """
        Initialize the MLP layers of the transformer.
        
        Parameters:
        - config (MLPConfig): The configuration for the MLP layers.
        """
        
        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        # Ensure hidden_dim is specified in the configuration
        assert config.hidden_dim is not None, "hidden_dim must be specified in the configuration."
        
        # Define the MLP layers
        # Define the dense layers
        # This will be lazily initialized in the forward pass, since we do not know the embedding size yet
        self.input_dense = Dense(config.hidden_dim, activation=ReLU()) # (B, S, E) -> (B, S, hidden_dim)
        self.output_dense: Dense  # (B, S, hidden_dim) -> (B, S, E)
        
        # Final dropout layer
        self.dropout: Dropout = Dropout(config.dropout) # (B, S, E) -> (B, S, E)
       
       
    ### Protected methods ### 
        
    def _forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MLP layers.
        
        Parameters:
        - x (Tensor): The input tensor.
        
        Returns:
        - Tensor: The output embeddings.
        """
        
        # Dimensions are:
        # - B: batch size
        # - S: sequence length
        # - E: embedding size (embedding dimension of the original data)
            
        # Apply the input dense layer to the data
        out = self.input_dense(x) # (B, S, E) -> (B, S, 4 * E)
        
        # Apply the output dense layer to the data
        return self.dropout(self.output_dense(out)) # (B, S, 4 * E) -> (B, S, E)
        
    
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
        
        # Unpack the shape of the input data
        _, _, E = x.shape # (B, S, E)  
        
        # Initialize the output dense layer
        self.output_dense = Dense(E) # (B, S, hidden_dim) -> (B, S, E)