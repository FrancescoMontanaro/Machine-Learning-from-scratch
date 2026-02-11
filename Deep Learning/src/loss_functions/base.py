from ..core import Tensor, ModuleOutput


class LossFn:    
    
    ### Magic methods ###

    def __call__(self, y_true: Tensor, y_pred: Tensor, **aux: Tensor) -> ModuleOutput:
        """
        Compute the loss.

        Parameters:
        - y_true (Tensor): True target variable
        - y_pred (Tensor): Predicted target variable
        - **aux (Tensor): Auxiliary tensors from ModuleOutput (e.g., mu, logvar for VAE)

        Returns:
        - ModuleOutput: the loss value as a ModuleOutput containing a single tensor
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method '__call__' is not implemented.")


    ### Public methods ###

    def step_epoch(self):
        """
        Optional method to be called at the end of each epoch during training.
        Can be used for tasks like KL annealing in VAEs.
        """
        
        # This method can be overridden by subclasses if needed. 
        # By default, it does nothing.
        pass