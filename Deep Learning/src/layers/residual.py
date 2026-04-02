from typing import Optional

from ..core import Tensor, Module
from ..activations import Activation


class ResidualBlock(Module):
    
    ### Magic methods ###
    
    def __init__(self, block: Module, activation: Optional[Activation] = None, *args, **kwargs) -> None:
        """
        Class constructor of the residual block layer
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Store the block and activation function
        self.block = block
        self.activation = activation
    
    
    ### Protected methods ###

    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Foward pass of the layer
        
        Parameters:
        - x (Tensor): Features of the dataset
        
        Returns:
        - Tensor: Output of the layer
        """

        # Apply the block to the input
        out = self.block(x, *args, **kwargs)

        # Add the residual connection
        out = out.output + x

        # Apply the activation function if provided
        if self.activation is not None:
            out = self.activation(out)

        # Return the output
        return out