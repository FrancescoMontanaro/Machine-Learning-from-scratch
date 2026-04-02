import numpy as np

from ...core import Tensor, Module, ModuleList, ModuleOutput
from .config import VAEConfig, VAEEncoderConfig, VAEDecoderConfig
from ...layers import Dense, Flatten, Conv2D, ConvTranspose2D, Reshape


class VAEEncoder(Module):

    ### Magic methods ###

    def __init__(self, latent_dim: int, encoder_config: VAEEncoderConfig, *args, **kwargs) -> None:
        """
        Initialize the Variational Autoencoder (VAE) Encoder architecture.

        Parameters:
        - latent_dim (int): Dimensionality of the latent space.
        - encoder_config (VAEEncoderConfig): Configuration for the VAE Encoder architecture.
        """

        # Initialize the superclass
        super().__init__(*args, **kwargs)

        # Save the configuration
        self.latent_dim = latent_dim
        
        # Shape before flatten (to be set during forward pass)
        self.pre_flatten_shape: tuple = ()

        # Initialize convolutional layers based on the provided configuration
        self.conv_layers = ModuleList([
            Conv2D(
                num_filters = conv_config.num_filters,
                kernel_size = conv_config.kernel_size,
                padding = conv_config.padding,
                stride = conv_config.stride,
                activation = conv_config.activation
            ) for conv_config in encoder_config.conv
        ])

        # Initialize the flatten layer
        self.flatten = Flatten()

        # Initialize the fully connected layers
        self.fc_layers = ModuleList([
            Dense(
                num_units = fc_config.num_units,
                activation = fc_config.activation
            ) for fc_config in encoder_config.fc
        ])

        self.fc_mu = Dense(self.latent_dim)
        self.fc_logvar = Dense(self.latent_dim)
        

    ### Protected methods ###

    def _forward(self, x: Tensor, *args, **kwargs) -> 'ModuleOutput':
        """
        Forward pass of the VAE Encoder.

        Parameters:
        - x (Tensor): Input tensor.

        Returns:
        - ModuleOutput: Output with z (reparameterized) as primary, mu and logvar as auxiliary.
        """
        
        # Forward pass through convolutional layers
        for conv in self.conv_layers:
            x = conv(x).output
        
        # Save shape before flatten (excluding batch dimension)
        self.pre_flatten_shape = x.shape[1:]
        
        # Flatten the convolutional output
        x = self.flatten(x).output

        # Forward pass through the fully connected layers
        for fc in self.fc_layers:
            x = fc(x).output

        # Compute the mean and log variance for the latent space
        mu = self.fc_mu(x).output
        logvar = self.fc_logvar(x).output

        # Reparameterization trick
        z = self._reparameterize(mu, logvar)

        # Return ModuleOutput with z as primary, mu and logvar as auxiliary
        return ModuleOutput(output=z, mu=mu, logvar=logvar)


    @staticmethod
    def _reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) using N(0,1).

        Parameters:
        - mu (Tensor): Mean tensor.
        - logvar (Tensor): Log variance tensor.

        Returns:
        - Tensor: Sampled tensor from N(mu, var).
        """
        
        # Compute standard deviation and sample epsilon
        std = (logvar * 0.5).exp()
        eps = Tensor.randn_like(std)

        # Return the sampled tensor
        return mu + std * eps
    

class VAEDecoder(Module):

    ### Magic methods ###

    def __init__(self, decoder_config: VAEDecoderConfig, *args, **kwargs) -> None:
        """
        Initialize the Variational Autoencoder (VAE) Decoder architecture.

        Parameters:
        - decoder_config (VAEDecoderConfig): Configuration for the VAE Decoder architecture.
        """

        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        # Save config for lazy initialization of fc layer
        self.decoder_config = decoder_config

        # Reshape layer will be initialized lazily
        self.reshape: Reshape # Will be set in set_reshape_shape()
        
        # FC layer will also be initialized lazily (needs reshape shape to compute units)
        self.fc = ModuleList([
            Dense(
                num_units = fc_config.num_units,
                activation = fc_config.activation
            ) for fc_config in decoder_config.fc
        ])

        # Initialize transposed convolutional layers based on the provided configuration
        self.deconv_layers = ModuleList([
            ConvTranspose2D(
                num_filters = deconv_config.num_filters,
                kernel_size = deconv_config.kernel_size,
                padding = deconv_config.padding,
                stride = deconv_config.stride,
                output_padding = deconv_config.output_padding,
                activation = deconv_config.activation
            ) for deconv_config in decoder_config.deconv
        ])
    
    
    ### Public methods ###
    
    def set_reshape_shape(self, shape: tuple) -> None:
        """
        Set the reshape shape and initialize the fc and reshape layers.
        
        Parameters:
        - shape (tuple): Shape to reshape to (H, W, C) - excluding batch dimension.
        """
        
        # Initialize the reshape layer
        self.reshape = Reshape(shape=shape)


    ### Protected methods ###

    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass of the VAE Decoder.

        Parameters:
        - x (Tensor): Latent space tensor.

        Returns:
        - Tensor: Reconstructed output tensor after passing through the decoder.
        """
        
        # Forward pass through the fully connected layers (if any)
        for fc in self.fc:
            x = fc(x).output

        # Forward pass through the reshape layer to get back to convolutional feature map shape
        x = self.reshape(x).output

        # Forward pass through transposed convolutional layers
        for deconv in self.deconv_layers:
            x = deconv(x).output

        # Return the reconstructed output
        return x
    

class VAE(Module):

    ### Magic methods ###

    def __init__(self, vae_config: VAEConfig, *args, **kwargs) -> None:
        """
        Initialize the Variational Autoencoder (VAE) architecture.
        
        Parameters:
        - vae_config (VAEConfig): Configuration for the VAE architecture.
        """

        # Initialize the parent class
        super().__init__(*args, **kwargs)

        # Create the encoder and decoder using the provided configurations
        self.encoder = VAEEncoder(
            latent_dim = vae_config.latent_dim,
            encoder_config = vae_config.encoder,
            name = "Encoder"
        )

        self.decoder = VAEDecoder(
            decoder_config = vae_config.decoder,
            name = "Decoder"
        )


    ### Protected methods ###

    def _forward(self, x: Tensor, *args, **kwargs) -> 'ModuleOutput':
        """
        Forward pass of the VAE architecture.
        
        Parameters:
        - x (Tensor): Input tensor.
        
        Returns:
        - ModuleOutput: Reconstructed output as primary, mu and logvar as auxiliary.
        """

        # Forward pass through the encoder
        enc = self.encoder(x)
        
        # Lazy initialization: set decoder reshape shape from encoder's pre-flatten shape
        if not self.decoder.is_initialized:
            self.decoder.set_reshape_shape(self.encoder.pre_flatten_shape)

        # Forward pass through the decoder (z is the primary output of encoder)
        recon = self.decoder(enc.output)

        # Return the reconstructed output with mu and logvar as auxiliary
        return ModuleOutput(output=recon.output, mu=enc.mu, logvar=enc.logvar)