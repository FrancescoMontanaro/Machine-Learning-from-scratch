import numpy as np
from typing import Optional, List, Tuple

from ..layers import Dropout
from ..activations import Tanh, Sigmoid
from ..core import Tensor, Module, ModuleOutput, TensorsList


class LSTM(Module):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        num_units: int,
        num_layers: int = 1,
        add_bias: bool = True, 
        dropout: float = 0.0,
        return_sequences: bool = False,
        *args, **kwargs
    ) -> None:
        """
        Class constructor for the LSTM layer
        
        Parameters:
        - num_units (int): Number of units in the layer (Dimensionality of the output space)
        - num_layers (int): Number of layers in the LSTM. Default is 1
        - add_bias (bool): Whether to include a bias term in the layer. Default is True
        - dropout (float): Dropout rate for the layer. Default is 0.0
        - return_sequences (bool): Whether to return the full sequence of outputs or just the last output. Default is False
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Store the configuration of the layer
        self.num_units = num_units
        self.num_layers = num_layers
        self.add_bias = add_bias
        self.dropout_rate = dropout
        self.return_sequences = return_sequences
        
        # Initialize the list of weights
        self.W: TensorsList = TensorsList()
        self.U: TensorsList = TensorsList()
        
        # Initialize biases if add_bias is True
        if self.add_bias:
            # Initialize the list of biases
            self.bias_ih: TensorsList = TensorsList()
            self.bias_hh: TensorsList = TensorsList()
            
        # Initialize the activation functions
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

        # Initialize the recurrent weights and biases
        self.dropout_layer = Dropout(self.dropout_rate) if self.dropout_rate > 0.0 else lambda x: x

    
    ### Protected methods ###
    
    def _forward(self, x: Tensor, h_prev: Optional[List[Tuple[Tensor, Tensor]]] = None, *args, **kwargs) -> Tensor:
        """
        Forward pass of the LSTM layer
        
        Parameters:
        - x (Tensor): Input data (Batch size, sequence length, embedding size)
        - h_prev (Optional[List[Tuple[Tensor]]]): Previous hidden states for each layer. Default is None
        
        Returns:
        - Tensor: Output of the layer
        """
        
        # Unpack the input shape
        batch_size, seq_length, _ = x.shape
        
        # If h_prev is None, initialize the hidden states for all layers to zeros
        if h_prev is None:
            # Initialize the hidden states to zeros
            h_t_prev = [
                Tensor(np.zeros((batch_size, self.num_units)), requires_grad=x.requires_grad)
                for _ in range(self.num_layers)
            ]
            
            # Initialize the cell states to zeros
            c_t_prev = [
                Tensor(np.zeros((batch_size, self.num_units)), requires_grad=x.requires_grad)
                for _ in range(self.num_layers)
            ]
            
        else:
            # Use the provided hidden states
            h_t_prev = [h[0] for h in h_prev]
            c_t_prev = [h[1] for h in h_prev]
                 
        # Add bias if specified
        if self.add_bias:            
            # Define the function to compute the gates with bias
            compute_gates = lambda x_t, i: (
                x_t @ self.W[i] + self.bias_ih[i] +
                h_t_prev[i] @ self.U[i] + self.bias_hh[i]
            )
            
        else:
            # Define the function to compute the gates without bias
            compute_gates = lambda x_t, i: (
                x_t @ self.W[i] + 
                h_t_prev[i] @ self.U[i]
            )

        # Create a list to store the outputs from the last layer for each time step
        outputs: list[Tensor] = []

        # Iterate over the sequence length
        for t in range(seq_length):
            # Extract the input for the current time step
            h_ti = x[:, t, :]
            
            # Create a list to store the hidden states and cell states for the current time step
            h_t: List[Tensor] = []
            c_t: List[Tensor] = []
            
            # Iterate over the number of layers
            for i in range(self.num_layers):
                # Compute the gates for the current time step and layer
                gates = compute_gates(h_ti.output if isinstance(h_ti, ModuleOutput) else h_ti, i)

                # Compute the input, forget, cell, and output gates
                i_t = self.sigmoid(gates[:, :self.num_units]) # Input gate
                f_t = self.sigmoid(gates[:, self.num_units:self.num_units*2]) # Forget gate
                g_t = self.tanh(gates[:, self.num_units*2:self.num_units*3]) # Cell gate
                o_t = self.sigmoid(gates[:, self.num_units*3:]) # Output gate
                
                # Compute the new cell state and hidden state
                c_ti = f_t * c_t_prev[i] + i_t * g_t
                h_ti = o_t * self.tanh(c_ti)
                    
                # Append the hidden state and cell state to the list
                h_t.append(h_ti)
                c_t.append(c_ti)
                
                # Apply dropout if specified and not the last layer
                if i < self.num_layers - 1:
                    # Apply dropout to the output of the current layer
                    h_ti = self.dropout_layer(h_ti)
                    
            # Update the hidden state and cell state for the current time step
            h_t_prev = h_t
            c_t_prev = c_t

            # Append the last layer's output for the current time step to the outputs list
            outputs.append(h_ti.output if isinstance(h_ti, ModuleOutput) else h_ti)
            
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
        assert len(x.shape) >= 2, f"Invalid input shape. Input must be at least a 2D array. Got shape: {x.shape}"
            
        ### Initialize the parameters for all the layers ###
        
        # Iterate through the number of layers
        for idx in range(self.num_layers):
            # Extract the number of features from the current layer 
            # and compute the standard deviation for weight initialization
            num_features = self.num_units if idx > 0 else x.shape[-1]
            std_dev = np.sqrt(1 / self.num_units) if idx > 0 else np.sqrt(1 / num_features)
            
            # Initialize the weights
            self.W.append(Tensor(
                data = np.random.uniform(-std_dev, std_dev, (num_features, self.num_units * 4)),
                requires_grad=True, is_parameter=True
            ))
            
            self.U.append(Tensor(
                data = np.random.uniform(-std_dev, std_dev, (self.num_units, self.num_units * 4)),
                requires_grad=True, is_parameter=True
            ))
            
            # Check if add_bias is enabled
            if self.add_bias:
                # Initialize the bias
                self.bias_ih.append(Tensor(
                    data = np.zeros(self.num_units * 4), 
                    requires_grad=True, is_parameter=True
                ))
                
                self.bias_hh.append(Tensor(
                    data = np.zeros(self.num_units * 4), 
                    requires_grad=True, is_parameter=True
                ))