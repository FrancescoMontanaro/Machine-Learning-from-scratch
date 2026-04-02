import numpy as np

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


    ### Protected methods ###

    @staticmethod
    def _prepare_classification_tensors(y_true: Tensor, y_pred: Tensor) -> tuple[Tensor, Tensor]:
        """
        Normalize targets/predictions for class-based losses (CE/KL style).

        Parameters:
        - y_true (Tensor): Target labels, either as class indices (N,) or distributions (N, C).
        - y_pred (Tensor): Predicted logits/probabilities, expected to have shape (N, C) or (N, T, C) for sequences.

        Returns:
        - tuple[Tensor, Tensor]: Normalized (y_true, y_pred) ready for loss computation, 
            where y_true is either class indices (N,) or distributions (N, C) and 
            y_pred is reshaped to (N, C) if it was a sequence.
        """

        # Sequence/class-axis case: flatten all non-class dimensions.
        if len(y_pred.shape) >= 3:
            # Extract the number of classes and reshape predictions to (N, C).
            num_classes = y_pred.shape[-1]
            base_shape = y_pred.shape[:-1]

            # Reshape predictions to (N, C) where N is the total number of tokens across the batch and sequence.
            y_pred = y_pred.reshape((-1, num_classes))

            # Reshape targets to match predictions. If y_true is already a distribution, reshape it to (N, C).
            if y_true.shape == (*base_shape, num_classes):
                y_true = y_true.reshape((-1, num_classes))
            elif y_true.shape == base_shape:
                y_true = y_true.reshape((-1,))
            else:
                raise ValueError(
                    f"Incompatible sequence shapes for class loss: y_true={y_true.shape}, y_pred={y_pred.shape}. "
                    "Expected y_true to match y_pred without class axis or to be a full distribution."
                )

            # Return the normalized tensors for classification loss.
            return y_true, y_pred

        # Last-step logits with sequence targets: use the last target token.
        if len(y_pred.shape) == 2 and len(y_true.shape) == 2 and y_true.shape != y_pred.shape:
            # If the batch sizes don't match, it's an error.
            if y_true.shape[0] != y_pred.shape[0]:
                raise ValueError(
                    f"Batch size mismatch for class loss: y_true={y_true.shape}, y_pred={y_pred.shape}."
                )
            
            # Use the last token in the sequence as the target class index.
            y_true = y_true[:, -1]

        # If y_true is already a distribution matching y_pred, return it as is.
        return y_true, y_pred


    @staticmethod
    def _to_class_distribution(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Convert class indices/distributions in y_true into a distribution matching y_pred.
        """

        # If y_true is already a distribution matching y_pred, return it as is.
        if y_true.shape == y_pred.shape:
            return y_true
        
        # If y_true is a 1D array of class indices, convert it to one-hot encoding.
        if len(y_pred.shape) != 2:
            raise ValueError(
                f"Class distribution conversion expects 2D predictions (N, C). Got y_pred={y_pred.shape}."
            )

        # If y_true is not 1D, it's an error (we only support class indices or distributions).
        if len(y_true.shape) != 1:
            raise ValueError(
                f"Expected class indices shape (N,) or distribution shape (N, C). Got y_true={y_true.shape}, y_pred={y_pred.shape}."
            )

        # Convert and return the one-hot encoded distribution.
        num_classes = y_pred.shape[-1]
        return Tensor(np.eye(num_classes)[y_true.data.astype(int)], requires_grad=False)
