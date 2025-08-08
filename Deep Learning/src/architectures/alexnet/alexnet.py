from .config import AlexNetConfig
from ...core import Tensor, Module
from ..sequential import Sequential
from ...layers import Dense, LocalResponseNormalization, Flatten, Dropout, MaxPool2D, Conv2D


class AlexNet(Sequential):
    
    ### Magic methods ###
    
    def __init__(self, config: AlexNetConfig = AlexNetConfig(), *args, **kwargs) -> None:
        """
        Initialize the AlexNet architecture.
        
        Parameters:
        - config (AlexNetConfig): Configuration for the AlexNet architecture.
        """
        
        # Initialize the superclass with the AlexNet core module
        super().__init__(
            modules = [AlexNetModule(config, name="AlexNet")],
            *args, **kwargs
        )


class AlexNetModule(Module):
    
    ### Magic methods ###
    
    def __init__(self, config: AlexNetConfig = AlexNetConfig(), *args, **kwargs) -> None:
        """
        Initialize the AlexNet architecture.
        
        Parameters:
        - config (AlexNetConfig): Configuration for the AlexNet architecture.
        
        Raises:
        - AssertionError: If the configuration does not meet the requirements for AlexNet.
        """
        
        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        ### Create the layers of the AlexNet architecture ###
        
        # Check if the configurations are valid
        assert len(config.conv_layers) == 5, "AlexNet requires exactly 5 convolutional layers."
        assert len(config.dense_layers) == 2, "AlexNet requires exactly 2 dense layers."
        assert config.local_response_norm is not None, "AlexNet requires a local response normalization configuration."
        assert config.max_pool is not None, "AlexNet requires a max pooling configuration."
        
        # Create the convolutional layers with the specified configurations
        self.conv_1 = Conv2D(
            num_filters = config.conv_layers[0].num_filters, 
            kernel_size = config.conv_layers[0].kernel_size, 
            stride = config.conv_layers[0].stride, 
            padding = config.conv_layers[0].padding, 
            activation = config.conv_layers[0].activation
        )
        
        self.conv_2 = Conv2D(
            num_filters = config.conv_layers[1].num_filters, 
            kernel_size = config.conv_layers[1].kernel_size, 
            stride = config.conv_layers[1].stride, 
            padding = config.conv_layers[1].padding, 
            activation = config.conv_layers[1].activation
        )
        
        self.conv_3 = Conv2D(
            num_filters = config.conv_layers[2].num_filters, 
            kernel_size = config.conv_layers[2].kernel_size, 
            stride = config.conv_layers[2].stride, 
            padding = config.conv_layers[2].padding, 
            activation = config.conv_layers[2].activation
        )
        
        self.conv_4 = Conv2D(
            num_filters = config.conv_layers[3].num_filters, 
            kernel_size = config.conv_layers[3].kernel_size, 
            stride = config.conv_layers[3].stride, 
            padding = config.conv_layers[3].padding, 
            activation = config.conv_layers[3].activation
        )
        
        self.conv_5 = Conv2D(
            num_filters = config.conv_layers[4].num_filters, 
            kernel_size = config.conv_layers[4].kernel_size, 
            stride = config.conv_layers[4].stride, 
            padding = config.conv_layers[4].padding, 
            activation = config.conv_layers[4].activation
        )
        
        # Create the normalization layers
        self.norm = LocalResponseNormalization(
            size = config.local_response_norm.size,
            alpha = config.local_response_norm.alpha,
            beta = config.local_response_norm.beta,
            k = config.local_response_norm.k
        )
        
        # Create the fully connected layers with the specified configurations
        self.fc_1 = Dense(
            num_units = config.dense_layers[0].num_units, 
            activation = config.dense_layers[0].activation
        )
        
        self.fc_2 = Dense(
            num_units = config.dense_layers[1].num_units, 
            activation = config.dense_layers[1].activation
        )
        
        self.fc_3 = Dense(num_units=config.num_classes, activation=None)
        
        # Create the flatten layer
        self.flatten = Flatten()
        
        # Create the dropout layer
        self.dropout = Dropout(config.dropout_rate)
        
        # Create the pooling layer
        self.max_pooling = MaxPool2D(
            size = config.max_pool.size, 
            stride = config.max_pool.stride, 
            padding = config.max_pool.padding
        )
        
        
    ### Protected methods ###
    
    def _forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the AlexNet architecture.
        
        Parameters:
        - x (Tensor): Input tensor of shape (B, H, W, C).
        
        Returns:
        - Tensor: Output tensor of shape (B, num_classes).
        """
        
        # Apply the convolutional layers with normalization and pooling
        x = self.max_pooling(self.norm(self.conv_1(x)))
        x = self.max_pooling(self.norm(self.conv_2(x)))
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.max_pooling(self.norm(self.conv_5(x)))
        
        # Flatten the output and apply the fully connected layers with dropout
        x = self.flatten(x)
        x = self.dropout(self.fc_1(x))
        x = self.dropout(self.fc_2(x))
        x = self.fc_3(x)
        
        # Return the output tensor
        return x