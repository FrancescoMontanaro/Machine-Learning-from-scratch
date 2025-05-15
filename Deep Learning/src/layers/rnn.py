import numpy as np
from typing import Optional, List

from ..layers import Dropout
from ..activations import Activation, Tanh
from ..core import Tensor, Module, TensorsList


class RNN(Module):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        num_units: int, 
        activation: Optional[Activation] = Tanh(), 
        add_bias: bool = True, 
        num_layers: int = 1,
        dropout: float = 0.0,
        return_sequences: bool = False,
        *args, **kwargs
    ) -> None:
        """
        Class constructor for the RNN layer
        
        Parameters:
        - num_units (int): Number of units in the layer
        - activation (Callable): Activation function of the layer. Default is Tanh
        - add_bias (bool): Whether to include a bias term in the layer. Default is True
        - num_layers (int): Number of layers in the RNN. Default is 1
        - dropout (float): Dropout rate for the layer. Default is 0.0
        - return_sequences (bool): Whether to return the full sequence of outputs or just the last output. Default is False
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Store the configuration of the layer
        self.activation = activation if activation is not None else lambda x: x
        self.num_units = num_units
        self.add_bias = add_bias
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.return_sequences = return_sequences
        
        # Initialize weights and biases as lists of Tensors
        self.weights: TensorsList = TensorsList()
        self.recurrent_weights: TensorsList = TensorsList()
        
        # Initialize biases if add_bias is True
        if self.add_bias:
            self.bias: TensorsList = TensorsList()
            self.recurrent_bias: TensorsList = TensorsList()

        # Initialize the recurrent weights and biases
        self.is_recurrent = True
        self.dropout_layer = Dropout(self.dropout_rate) if self.dropout_rate > 0.0 else lambda x: x

    
    ### Protected methods ###
    
    def _forward(self, x: Tensor, h_prev: Optional[List[Tensor]] = None) -> Tensor:
        """
        Forward pass of the RNN layer
        
        Parameters:
        - x (Tensor): Input data (Batch size, sequence length, embedding size)
        - h_prev (Optional[List[Tensor]]): Previous hidden states for all layers. Each element is (Batch size, num_units).
        
        Returns:
        - Tensor: Output of the layer
        """
        
        # Unpack the input shape
        batch_size, seq_length, _ = x.shape()
        
        # If h_prev is None, initialize the hidden states for all layers to zeros
        if h_prev is None:
            # Initialize the hidden states to zeros
            hidden_states_prev_t = [
                Tensor(np.zeros((batch_size, self.num_units)), requires_grad=x.requires_grad)
                for _ in range(self.num_layers)
            ]
        else:
            # Use the provided hidden states
            hidden_states_prev_t = h_prev
            
        # Add bias if specified
        if self.add_bias:
            # Define the function to compute the hidden state with bias
            compute_hidden = lambda x_t, i: (
                x_t @ self.weights[i] + self.bias[i] +
                hidden_states_prev_t[i] @ self.recurrent_weights[i] + self.recurrent_bias[i]
            )
            
        else:
            # Define the function to compute the hidden state without bias
            compute_hidden = lambda x_t, i: (
                x_t @ self.weights[i] +
                hidden_states_prev_t[i] @ self.recurrent_weights[i]
            )

        # Create a list to store the outputs from the last layer for each time step
        outputs: list[Tensor] = []

        # Iterate over the sequence length
        for t in range(seq_length):
            # Extract the input for the current time step
            h_ti = x[:, t, :]
            
            # Create a list to store the hidden states for the current time step
            h_t: List[Tensor] = []

            # Iterate over the number of layers
            for i in range(self.num_layers):
                # Compute the hidden state for the current time step and layer
                h_ti = self.activation(compute_hidden(h_ti, i))

                # Append the hidden state to the list without applying dropout
                h_t.append(h_ti)

                # Apply dropout if specified and not the last layer
                if i < self.num_layers - 1:
                    # Apply dropout to the output of the current layer
                    h_ti = self.dropout_layer(h_ti)

            # Update the hidden state for the current time step and the previous time one
            hidden_states_prev_t = h_t

            # Append the last layer's output for the current time step to the outputs list
            outputs.append(h_ti)
            
        # Stack the outputs along the time dimension
        out = Tensor.stack(outputs, axis=1)

        # If the architecture is not returning the sequence, return only the last output
        if not self.return_sequences:
            # Return the last output
            return out[:, -1, :]
            
        # Return the output of the architecture
        return out

    
    def _lazy_init(self, x: Tensor, *args, **kwargs) -> None:
        """
        Lazy initialization of the parameters of the layer
        
        Parameters:
        - x (Tensor): Input data
        
        Raises:
        - AssertionError: If the input shape is invalid
        """
        
        # Check if the input shape is valid
        assert len(x.shape()) >= 2, f"Invalid input shape. Input must be at least a 2D array. Got shape: {x.shape()}"
        
        ### Initialize the input weights and biases ###
        
        # Extract the number of features from the input data 
        # and compute the standard deviation for weight initialization
        num_features = x.shape()[-1]
        std_dev = np.sqrt(1 / num_features)
        
        # Initialize the weights
        self.weights.append(Tensor(
            data = np.random.uniform(-std_dev, std_dev, (num_features, self.num_units)),
            requires_grad=True, is_parameter=True
        ))
        
        # Initialize the recurrent weights
        self.recurrent_weights.append(Tensor(
            data = np.random.uniform(-std_dev, std_dev, (self.num_units, self.num_units)),
            requires_grad=True, is_parameter=True
        ))
        
        # Check if add_bias is enabled
        if self.add_bias:
            # Initialize the bias
            self.bias.append(Tensor(
                data = np.zeros(self.num_units), 
                requires_grad=True, is_parameter=True
            ))
            
            # Initialize the recurrent bias
            self.recurrent_bias.append(Tensor(
                data = np.zeros(self.num_units), 
                requires_grad=True, is_parameter=True
            ))
            
        ### Initialize the parameters for the remaining layers ###
        
        # Iterate through the number of layers
        for _ in range(1, self.num_layers):
            # Extract the number of features from the current layer 
            # and compute the standard deviation for weight initialization
            num_features = self.num_units
            std_dev = np.sqrt(1 / self.num_units)
            
            # Initialize the weights
            self.weights.append(Tensor(
                data = np.random.uniform(-std_dev, std_dev, (num_features, self.num_units)),
                requires_grad=True, is_parameter=True
            ))
            
            # Initialize the recurrent weights
            self.recurrent_weights.append(Tensor(
                data = np.random.uniform(-std_dev, std_dev, (self.num_units, self.num_units)),
                requires_grad=True, is_parameter=True
            ))
            
            # Check if add_bias is enabled
            if self.add_bias:
                # Initialize the bias
                self.bias.append(Tensor(
                    data = np.zeros(self.num_units), 
                    requires_grad=True, is_parameter=True
                ))
                
                # Initialize the recurrent bias
                self.recurrent_bias.append(Tensor(
                    data = np.zeros(self.num_units), 
                    requires_grad=True, is_parameter=True
                ))