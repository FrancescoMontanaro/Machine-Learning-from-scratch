import numpy as np

from ..core import Tensor, Module


class Embedding(Module):
    
    ### Magic methods ###
    
    def __init__(self, input_dim: int, output_dim: int, *args, **kwargs) -> None:
        """
        Class constructor
        
        Parameters:
        - input_dim (int): Number of features in the input data (e.g. vocabulary size)
        - output_dim (int): Embedding dimension
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Store the activation function
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize the Embedding matrix
        self.embedding: Tensor
    
    
    ### Protected methods ###

    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Foward pass of the layer
        
        Parameters:
        - x (Tensor): Features of the dataset
        
        Returns:
        - Tensor: Output of the layer
        """
        
        # If the input is 1D, reshape it to 2D
        if len(x.shape) == 1:
            # reshape the input to 2D
            x = x.reshape((1, -1))
            
            # Compute the embedding
            out = self.embedding[x.data.astype(int)]
            
            # Squeeze the output to 1D
            return out.squeeze(0)
    
        # Return the embeddings
        return self.embedding[x.data.astype(int)]
    
        
    def _lazy_init(self, x: Tensor, *args, **kwargs) -> None:
        """
        Method to initialize the parameters of the layer
        
        Parameters:
        - x (Tensor): Features of the dataset
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape) <= 2, f"Invalid input shape. Input must be at maximum a 2D array. Got shape: {x.shape}"
        
        # Initialize the embedding matrix
        self.embedding = Tensor(
            data = np.random.uniform(-1, 1, (self.input_dim, self.output_dim)),
            requires_grad = True,
            is_parameter = True
        )