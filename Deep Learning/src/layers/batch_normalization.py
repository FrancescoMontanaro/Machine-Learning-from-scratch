import numpy as np

from ..core import Tensor, Module


class BatchNormalization(Module):
    
    ### Magic methods ###
    
    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.1, *args, **kwargs) -> None:
        """
        Initialize the batch normalization layer.
        
        Parameters:
        - epsilon (float): The epsilon parameter for numerical stability.
        - momentum (float): The momentum parameter for the moving average.
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Initialize the epsilon parameter
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Initialize the scale and shift parameters
        self.gamma: Tensor
        self.beta: Tensor
        
        # Initialize the running mean and variance
        self.running_mean: Tensor
        self.running_var: Tensor

     
    ### Protected methods ###
    
    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass of the batch normalization layer.
        
        Parameters:
        - x (Tensor): The input tensor.
        
        Returns:
        - Tensor: The normalized tensor.
        """
        
        # The layer is in training phase
        if self.training:
            # Determine axes to compute mean and variance: all except the last dimension (features)
            axes = tuple(range(len(x.shape()) - 1))
            
            # Calculate batch mean and variance
            mean = x.mean(axis=axes, keepdims=True)
            var = x.var(axis=axes, keepdims=True, ddof=0)
            
            # Update running mean and variance
            self.running_mean.data = self.running_mean.data * (1 - self.momentum) + mean.data * self.momentum
            self.running_var.data = self.running_var.data * (1 - self.momentum) + var.data * self.momentum
            
            # Normalize and return the output
            return (x - mean) / (var + self.epsilon).sqrt() * self.gamma + self.beta
           
        # The layer is in inference phase 
        else:
            # Use running statistics for normalization
            return self.gamma * ((x - self.running_mean) * (1 / (self.running_var + self.epsilon).sqrt())) + self.beta
    
    
    def _lazy_init(self, x: Tensor, *args, **kwargs) -> None:
        """
        Method to initialize the parameters of the layer
        
        Raises:
        - AssertionError: If the input shape is not set
        """
        
        # Extract the shape of the parameters
        num_features = x.shape()[-1]
        
        # Initialize the scale parameter
        self.gamma = Tensor(
            data = np.ones(num_features),
            requires_grad = True,
            is_parameter = True
        )
        
        # Initialize the shift parameter
        self.beta = Tensor(
            data = np.zeros(num_features),
            requires_grad = True,
            is_parameter = True
        )
        
        # Initialize the running mean and variance
        running_mean = Tensor(np.zeros(num_features), requires_grad=False, is_parameter=False)
        running_var = Tensor(np.ones(num_features), requires_grad=False, is_parameter=False)
        
        # Store the running mean and variance as buffers
        self.register_buffer("running_mean", running_mean)
        self.register_buffer("running_var", running_var)