import numpy as np
from typing import Optional

from ..core import Tensor, Module
from ..activations import Activation


class Dense(Module):
    
    ### Magic methods ###
    
    def __init__(self, num_units: int, activation: Optional[Activation] = None, name: Optional[str] = None) -> None:
        """
        Class constructor
        
        Parameters:
        - num_units (int): Number of units in the layer
        - activation (Callable): Activation function of the layer. Default is ReLU
        - name (str): Name of the layer
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Store the activation function
        self.activation = activation
        self.num_units = num_units
        
        # Initialize the weights and bias
        self.weights: Tensor
        self.bias: Tensor
    
    
    ### Public methods ###

    def forward(self, x: Tensor) -> Tensor:
        """
        Foward pass of the layer
        
        Parameters:
        - x (Tensor): Features of the dataset
        
        Returns:
        - Tensor: Output of the layer
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        - AssertionError: If the weights and bias are not initialized
        """
        
        # Check if the input shape is valid
        assert len(x.shape()) == 2, f"Invalid input shape. Input must be a 2D array. The shape must be (Batch size, number of features). Got shape: {x.shape()}"
        
        # Unpack the shape of the input data for better readability
        batch_size, num_features = x.shape()
        
        # Store the input shape
        self.input_shape = (batch_size, num_features)
        
        # Check if the layer is initialized
        if not self.initialized:
            # Initialize the layer
            self.init_params(num_features)
        
        # Assert that the weights and bias are initialized
        assert isinstance(self.weights, Tensor), "Weights are not initialized. Please call the layer with some input data to initialize the weights."
        assert isinstance(self.bias, Tensor), "Bias is not initialized. Please call the layer with some input data to initialize the bias."
        
        # Compute the linear combination of the weights and features
        self.linear_comb = x @ self.weights + self.bias
        
        # Return the output of the neuron
        return self.activation(self.linear_comb) if self.activation is not None else self.linear_comb
    
    
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tuple: The shape of the output of the layer
        """
        
        # Call the parent class method to check if the layer is initialized
        super().output_shape()
        
        # Unpack the input shape for better readability
        batch_size, _ = self.input_shape
        
        # The output shape
        return (batch_size, self.num_units) # (Batch size, number of units)
    
        
    def init_params(self, num_features: int) -> None:
        """
        Method to initialize the weights and bias of the layer
        
        Parameters:
        - num_features (int): Number of features in the input data
        """
        
        # Initialize the weights with random values
        self.weights = Tensor(
            data = np.random.uniform(-np.sqrt(1 / num_features), np.sqrt(1 / num_features), (num_features, self.num_units)),
            requires_grad = True,
            is_parameter = True
        )
        
        # Initialize the bias with zeros
        self.bias = Tensor(
            data = np.zeros(self.num_units),
            requires_grad = True,
            is_parameter = True
        )
        
        # Call the parent class method to set the layer as initialized
        super().init_params()