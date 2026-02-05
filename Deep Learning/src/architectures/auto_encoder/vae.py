import numpy as np
from typing import Tuple

from ...core import Tensor, MultiOutputModule, SingleOutputModule
from .config import VAEConfig, VAEEncoderConfig, VAEDecoderConfig
from ...layers import Dense, Flatten, Conv2D, ConvTranspose2D, Reshape


class VAEEncoder(MultiOutputModule):

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

        # Initialize layers based on the configuration
        self.conv1 = Conv2D(
            num_filters = encoder_config.conv_1.num_filters,
            kernel_size = encoder_config.conv_1.kernel_size,
            padding = encoder_config.conv_1.padding,
            stride = encoder_config.conv_1.stride,
            activation = encoder_config.conv_1.activation
        )

        self.conv2 = Conv2D(
            num_filters = encoder_config.conv_2.num_filters,
            kernel_size = encoder_config.conv_2.kernel_size,
            padding = encoder_config.conv_2.padding,
            stride = encoder_config.conv_2.stride,
            activation = encoder_config.conv_2.activation
        )

        self.flatten = Flatten()

        self.fc = Dense(
            num_units = encoder_config.fc.num_units,
            activation = encoder_config.fc.activation
        )

        self.fc_mu = Dense(self.latent_dim)
        self.fc_logvar = Dense(self.latent_dim)
        

    ### Protected methods ###

    def _forward(self, x: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the VAE Encoder.

        Parameters:
        - x (Tensor): Input tensor.

        Returns:
        - Tuple[Tensor, Tensor]: Mean and log variance tensors after passing through the encoder.
        """
        
        # Forward pass through the layers
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Save shape before flatten (excluding batch dimension)
        self.pre_flatten_shape = x.shape[1:]
        
        x = self.flatten(x)
        x = self.fc(x)

        # Compute the mean and log variance for the latent space
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Return the mean and log variance
        return mu, logvar
    

class VAEDecoder(SingleOutputModule):

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
        self.fc: Dense # Will be set in set_reshape_shape()

        self.deconv_1 = ConvTranspose2D(
            num_filters = decoder_config.deconv_1.num_filters,
            kernel_size = decoder_config.deconv_1.kernel_size,
            padding = decoder_config.deconv_1.padding,
            stride = decoder_config.deconv_1.stride,
            output_padding = decoder_config.deconv_1.output_padding,
            activation = decoder_config.deconv_1.activation
        )

        self.deconv_2 = ConvTranspose2D(
            num_filters = decoder_config.deconv_2.num_filters,
            kernel_size = decoder_config.deconv_2.kernel_size,
            padding = decoder_config.deconv_2.padding,
            stride = decoder_config.deconv_2.stride,
            output_padding = decoder_config.deconv_2.output_padding,
            activation = decoder_config.deconv_2.activation
        )
        
        self.deconv_3 = ConvTranspose2D(
            num_filters = decoder_config.deconv_3.num_filters,
            kernel_size = decoder_config.deconv_3.kernel_size,
            padding = decoder_config.deconv_3.padding,
            stride = decoder_config.deconv_3.stride,
            output_padding = decoder_config.deconv_3.output_padding,
            activation = decoder_config.deconv_3.activation
        )
    
    
    ### Public methods ###
    
    def set_reshape_shape(self, shape: tuple) -> None:
        """
        Set the reshape shape and initialize the fc and reshape layers.
        
        Parameters:
        - shape (tuple): Shape to reshape to (H, W, C) - excluding batch dimension.
        """
        
        # Calculate the number of units for the fc layer
        fc_units = int(np.prod(shape))
        
        # Initialize the fc layer with the correct number of units
        self.fc = Dense(
            num_units = fc_units, 
            activation = self.decoder_config.fc.activation
        )
        
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
        
        # Forward pass through the layers
        x = self.fc(x)
        x = self.reshape(x)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)

        # Return the reconstructed output
        return x
    

class VAE(MultiOutputModule):

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

    def _forward(self, x: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the VAE architecture.
        
        Parameters:
        - x (Tensor): Input tensor.
        
        Returns:
        - Tuple[Tensor, Tensor, Tensor]: Reconstructed output, mean, and log variance tensors.
        """

        # Forward pass through the encoder
        mu, logvar = self.encoder(x)
        
        # Lazy initialization: set decoder reshape shape from encoder's pre-flatten shape
        if not self.decoder.is_initialized:
            self.decoder.set_reshape_shape(self.encoder.pre_flatten_shape)

        # Reparameterization trick to sample from N(mu, var) from N(0,1) in a differentiable way
        z = self._reparameterize(mu, logvar)

        # Forward pass through the decoder
        recon = self.decoder(z)

        # Return the reconstructed output, mean, and log variance
        return recon, mu, logvar


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