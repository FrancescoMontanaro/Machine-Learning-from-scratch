import numpy as np
from typing import Optional

from ..core import Tensor, Module
from ..activations import Activation


class Dense(Module):
    
    ### Magic methods ###
    
    def __init__(self, num_units: int, activation: Optional[Activation] = None, add_bias: bool = True, *args, **kwargs) -> None:
        """
        Class constructor
        
        Parameters:
        - num_units (int): Number of units in the layer
        - activation (Callable): Activation function of the layer. Default is ReLU
        - add_bias (bool): Whether to include a bias term in the layer. Default is True
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Store the activation function
        self.activation = activation
        self.num_units = num_units
        self.add_bias = add_bias
        
        # Initialize the weights and bias
        self.weights: Tensor
        self.bias: Tensor
    
    
    ### Public methods ###

    def _forward(self, x: Tensor) -> Tensor:
        """
        Foward pass of the layer
        
        Parameters:
        - x (Tensor): Features of the dataset
        
        Returns:
        - Tensor: Output of the layer
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Store the input shape
        input_shape = x.shape()
        
        # The input data is 2D
        if len(input_shape) == 2:
            # Compute the linear combination of the weights and features
            linear_comb = x @ self.weights
            
            # Add the bias term if necessary
            if self.add_bias:
                linear_comb += self.bias
            
            # Return the output of the neuron
            return self.activation(linear_comb) if self.activation is not None else linear_comb
        
        # The input data greater than 2D
        else:
            # Extract the original shape of the input data
            original_shape = input_shape
            
            # Flatten the input data in the last dimension
            x_flat = x.reshape((-1, original_shape[-1]))
            
            # Apply the linear transformation
            linear_comb = x_flat @ self.weights
            
            # Add the bias term if necessary
            if self.add_bias:
                linear_comb += self.bias
            
            # Apply the activation function
            out_flat = self.activation(linear_comb) if self.activation is not None else linear_comb
            
            # Reshape the output to the original shape
            new_shape = original_shape[:-1] + (self.num_units,)
            
            # Return the output of the layer
            return out_flat.reshape(new_shape)
    
        
    def _lazy_init(self, x: Tensor) -> None:
        """
        Method to initialize the weights and bias of the layer
        
        Parameters:
        - x (Tensor): Features of the dataset
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape()) >= 2, f"Invalid input shape. Input must be at least a 2D array. Got shape: {x.shape()}"
        
        # Extract the number of features from the input data
        num_features = x.shape()[-1]
        
        # Initialize the weights with random values
        self.weights = Tensor(
            data = np.random.uniform(-np.sqrt(1 / num_features), np.sqrt(1 / num_features), (num_features, self.num_units)),
            requires_grad = True,
            is_parameter = True
        )
        
        # Initialize the bias with zeros if necessary
        if self.add_bias:
            self.bias = Tensor(
                data = np.zeros(self.num_units),
                requires_grad = True,
                is_parameter = True
            )