from .vae import VAEEncoder, VAEDecoder
from ...core import Tensor, Module, ModuleOutput
from .config import VAEConfig, VAEEncoderConfig, VAEDecoderConfig


class ConditionalVAEEncoder(VAEEncoder):

    ### Magic methods ###

    def __init__(self, latent_dim: int, num_classes: int, encoder_config: VAEEncoderConfig, *args, **kwargs) -> None:
        """
        Initialize the Conditional Variational Autoencoder (CVAE) Encoder architecture.

        Parameters:
        - latent_dim (int): Dimensionality of the latent space.
        - num_classes (int): Number of classes for conditioning.
        - encoder_config (VAEEncoderConfig): Configuration for the VAE Encoder architecture.
        """

        # Initialize the superclass
        super().__init__(latent_dim=latent_dim, encoder_config=encoder_config, *args, **kwargs)

        # Save the configuration
        self.num_classes = num_classes
        

    ### Protected methods ###

    def _forward(self, x: Tensor, y: Tensor, *args, **kwargs) -> 'ModuleOutput':
        """
        Forward pass of the CVAE Encoder.

        Parameters:
        - x (Tensor): Input tensor.
        - y (Tensor): One-hot encoded class labels for conditioning.

        Returns:
        - ModuleOutput: Output with z (reparameterized) as primary, mu and logvar as auxiliary.
        """
        
        # Forward pass through the layers
        x = self.conv1(x).output
        x = self.conv2(x).output
        
        # Save shape before flatten (excluding batch dimension)
        self.pre_flatten_shape = x.shape[1:]
        
        # Flatten the convolutional output
        x = self.flatten(x).output

        # Concatenate the flattened output with the class labels
        x = Tensor.concat([x, y], axis=1)

        # Forward pass through the fully connected layer
        x = self.fc(x).output

        # Compute the mean and log variance for the latent space
        mu = self.fc_mu(x).output
        logvar = self.fc_logvar(x).output

        # Reparameterization trick
        z = self._reparameterize(mu, logvar)

        # Return ModuleOutput with z as primary, mu and logvar as auxiliary
        return ModuleOutput(output=z, mu=mu, logvar=logvar)
    

class ConditionalVAEDecoder(VAEDecoder):

    ### Magic methods ###

    def __init__(self, num_classes: int, decoder_config: VAEDecoderConfig, *args, **kwargs) -> None:
        """
        Initialize the Conditional Variational Autoencoder (CVAE) Decoder architecture.

        Parameters:
        - num_classes (int): Number of classes for conditioning.
        - decoder_config (VAEDecoderConfig): Configuration for the CVAE Decoder architecture.
        """

        # Initialize the superclass
        super().__init__(decoder_config=decoder_config, *args, **kwargs)
        
        # Save the configuration
        self.num_classes = num_classes


    ### Protected methods ###

    def _forward(self, x: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass of the CVAE Decoder.

        Parameters:
        - x (Tensor): Latent space tensor.
        - y (Tensor): One-hot encoded class labels for conditioning.

        Returns:
        - Tensor: Reconstructed output tensor after passing through the decoder.
        """

        # Concatenate the latent space tensor with the class labels
        z = Tensor.concat([x, y], axis=1)
        
        # Forward pass through the layers
        x = self.fc(z).output
        x = self.reshape(x).output
        x = self.deconv_1(x).output
        x = self.deconv_2(x).output
        x = self.deconv_3(x).output

        # Return the reconstructed output
        return x
    

class ConditionalVAE(Module):

    ### Magic methods ###

    def __init__(self, vae_config: VAEConfig, num_classes: int, *args, **kwargs) -> None:
        """
        Initialize the Conditional Variational Autoencoder (CVAE) architecture.
        
        Parameters:
        - vae_config (VAEConfig): Configuration for the CVAE architecture.
        - num_classes (int): Number of classes for conditioning.
        """

        # Initialize the parent class
        super().__init__(*args, **kwargs)

        # Create the encoder and decoder using the provided configurations
        self.encoder = ConditionalVAEEncoder(
            latent_dim = vae_config.latent_dim,
            encoder_config = vae_config.encoder,
            num_classes = num_classes,
            name = "Encoder"
        )

        self.decoder = ConditionalVAEDecoder(
            num_classes = num_classes,
            decoder_config = vae_config.decoder,
            name = "Decoder"
        )


    ### Protected methods ###

    def _forward(self, x: Tensor, y: Tensor, *args, **kwargs) -> 'ModuleOutput':
        """
        Forward pass of the CVAE architecture.
        
        Parameters:
        - x (Tensor): Input tensor.
        - y (Tensor): One-hot encoded class labels for conditioning.
        
        Returns:
        - ModuleOutput: Reconstructed output as primary, mu and logvar as auxiliary.
        """

        # Forward pass through the encoder
        enc = self.encoder(x=x, y=y)
        
        # Lazy initialization: set decoder reshape shape from encoder's pre-flatten shape
        if not self.decoder.is_initialized:
            self.decoder.set_reshape_shape(self.encoder.pre_flatten_shape)

        # Forward pass through the decoder (z is the primary output of encoder)
        recon = self.decoder(x=enc.output, y=y)

        # Return the reconstructed output with mu and logvar as auxiliary
        return ModuleOutput(output=recon.output, mu=enc.mu, logvar=enc.logvar)