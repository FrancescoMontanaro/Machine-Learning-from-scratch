from ..core import Tensor, Module

class LocalResponseNormalization(Module):
    
    ### Magic methods ###
    
    def __init__(self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0, *args, **kwargs) -> None:
        """
        Initialize the Local Response Normalization layer.
        
        Parameters:
        - size (int): size of the normalization window
        - alpha (float): scaling factor for the squared sum
        - beta (float): exponent for the normalization
        - k (float): constant added to the denominator to avoid division by zero
        """
        
        # Call the parent constructor
        super().__init__(*args, **kwargs)
        
        # Validate parameters
        if size < 1:
            # Raise an error if size is invalid
            raise ValueError("size must be >= 1")
        
        # Store parameters
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        
        # Half size for the normalization window
        self.half = self.size // 2
        
    
    ### Protected methods ###

    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Function to compute the forward pass of the Local Response Normalization layer.
        
        Parameters:
        - x (Tensor): input data
        
        Returns:
        - Tensor: output data after normalization
        """
        
        # Extract the number of channels from the input tensor
        _, _, _, C = x.shape()

        # Compute the squared values of the input tensor
        x2 = x * x 

        # Initialize a list to hold the parts of the squared sum
        parts = []
        
        # Loop over each channel to compute the local response normalization
        for c in range(C):
            # Compute the start and end indices for the normalization window
            start = max(0, c - self.half)
            end = min(C, c + self.half + 1)
            
            # Sum the squared values over the normalization window
            win_sum = x2[..., start:end].sum(axis=-1, keepdims=True)
            
            # Append the computed sum to the parts list
            parts.append(win_sum)

        # Concatenate the parts to reconstruct the channel axis
        squared_sum: Tensor = parts[0]
        for p in parts[1:]:
            # Concatenate along the channel axis
            squared_sum = squared_sum.concat([p], axis=-1)

        # Compute the normalization denominator
        denom = (self.k + (self.alpha / self.size) * squared_sum) ** self.beta

        # Return the normalized output
        return x / denom