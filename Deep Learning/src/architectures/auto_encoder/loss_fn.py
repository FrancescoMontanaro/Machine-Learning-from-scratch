from ...core import Tensor
from ...loss_functions import LossFn, BinaryCrossEntropy


class VAELoss(LossFn):
            
    ### Magic methods ###

    def __init__(
        self, 
        from_logits: bool = False, 
        beta: float = 1.0, 
        normalize_reconstruction: bool = True,
        annealing_epochs: int = 0
    ) -> None:
        """
        Initialize the VAE loss function, which combines binary cross-entropy and KL divergence.
        
        Parameters:
        - from_logits (bool): Whether the input is logits or probabilities for the BCE loss. Default is False
        - beta (float): Weight for the KL divergence term (beta-VAE). Lower values prioritize reconstruction.
                       Default is 1.0. Values like 0.1 or 0.01 can help if the model struggles to learn.
        - normalize_reconstruction (bool): If True, normalize BCE by number of pixels (mean instead of sum).
                                          This gives loss values in ~0.1-0.5 range. Default is True.
        - annealing_epochs (int): Number of epochs over which to linearly anneal beta from 0 to its target value.
                                 If 0, no annealing is applied (beta is constant). Default is 0.
        """
        
        # Initialize the BCE loss function
        self.bce_loss = BinaryCrossEntropy(from_logits=from_logits, reduction=None)
        
        # Store beta weight for KL divergence
        self.target_beta = beta  # Final beta value after annealing
        self.beta = 0.0 if annealing_epochs > 0 else beta  # Start from 0 if annealing
        
        # Store normalization flag
        self.normalize_reconstruction = normalize_reconstruction
        
        # Annealing configuration
        self.annealing_epochs = annealing_epochs
        self.current_epoch = 0
    

    def __call__(self, y_true: Tensor, y_pred: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Compute the VAE loss, which is the sum of the reconstruction loss (BCE) and the weighted KL divergence.
        
        Parameters:
        - y_true (Tensor): Original input tensor.
        - y_pred (Tensor): Reconstructed tensor from the VAE.
        - mu (Tensor): Mean tensor from the encoder's latent space.
        - logvar (Tensor): Log variance tensor from the encoder's latent space.
        
        Returns:
        - Tensor: the VAE loss (reconstruction + beta * KL) value as a tensor
        """
        
        # Compute reconstruction loss using binary cross-entropy
        bce = self.bce_loss(y_true, y_pred)
        
        if self.normalize_reconstruction:
            # Mean over all dimensions
            rec = bce.mean()
        else:
            # Sum over spatial dimensions, mean over batch
            rec = bce.sum(axis=(1,2,3)).mean()

        # Compute KL divergence for VAE: KL(q(z|x) || N(0,1))
        kl = (logvar.exp() + mu ** 2 - 1 - logvar) * 0.5
        kl = kl.sum(axis=1).mean()
        
        # Normalize KL by latent dimension if normalizing reconstruction (for balanced scaling)
        if self.normalize_reconstruction:
            latent_dim = mu.shape[-1]
            kl = kl / latent_dim

        # Compute total loss with beta weighting on KL term
        return rec + self.beta * kl


    ### Public methods ###

    def step_epoch(self) -> None:
        """
        Update the beta value based on the current epoch for KL annealing.
        Call this method at the end of each epoch during training.
        """
        
        # Update beta if annealing is enabled
        if self.annealing_epochs > 0:
            # Increment current epoch
            self.current_epoch += 1

            # Linear annealing from 0 to target_beta over annealing_epochs
            progress = min(self.current_epoch / self.annealing_epochs, 1.0)

            # Update beta value
            self.beta = self.target_beta * progress