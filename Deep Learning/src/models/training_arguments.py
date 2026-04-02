from typing import Tuple
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, TYPE_CHECKING

from ..core import Tensor
from ..optimizers import Optimizer
from ..loss_functions import LossFn
if TYPE_CHECKING: from .data_loader import DataLoader


@dataclass
class LabeledData:
    """
    Container for input data and corresponding target labels.
    
    Parameters:
    - input (Dict[str, Tensor]): Dictionary of named input tensors
    - target (Tensor): Target tensor (labels)

    Properties:
    - input_tuple (Tuple[Tensor, ...]): Tuple of input tensors extracted from the input dictionary
    - input_keys (Tuple[str, ...]): Tuple of keys corresponding to the input tensors
    """
    
    input: Dict[str, Tensor]
    target: Tensor

    @property
    def input_tuple(self) -> Tuple[Tensor, ...]:
        """
        Convert the input dictionary to a tuple of tensors.
        
        Returns:
            Tuple[Tensor, ...]: A tuple containing the input tensors in the order they were defined in the dictionary.
        """

        # Return the input tensors as a tuple, preserving the order of the dictionary values.
        return tuple(self.input.values())
    

    @property
    def input_keys(self) -> Tuple[str, ...]:
        """
        Get the keys of the input dictionary as a tuple.
        
        Returns:
            Tuple[str, ...]: A tuple containing the keys of the input dictionary.
        """

        # Return the keys of the input dictionary as a tuple.
        return tuple(self.input.keys())


@dataclass
class TrainingArguments:
    """
    Configuration for training parameters.
    
    Parameters:
    - data_loader (DataLoader): DataLoader instance containing training and validation data
    - optimizer (Optimizer): Optimizer to use for training
    - loss_fn (LossFn): Loss function to use for training
    - train_batch_size (int): Number of samples per training batch (default: 32)
    - eval_batch_size (Optional[int]): Number of samples per evaluation batch (default: None, uses train_batch_size)
    - num_epochs (int): Number of epochs to train the model (default: 10)
    - shuffle (bool): Whether to shuffle the training data between epochs (default: True)
    - gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating the parameters (default: 1)
    - metrics (list[Callable[..., Tensor]]): List of metrics to compute during training (default: [])
    - callbacks (list[Callable]): List of callbacks to execute during training (default: [])
    """
    
    # Mandatory parameters - Data
    data_loader: "DataLoader"
    
    # Mandatory parameters - Training
    optimizer: Optimizer
    loss_fn: LossFn
    
    # Optional parameters - Training configuration
    train_batch_size: int = 32
    eval_batch_size: Optional[int] = None
    num_epochs: int = 10
    shuffle: bool = True
    gradient_accumulation_steps: int = 1
    metrics: list[Callable[..., Tensor]] = field(default_factory=list)
    callbacks: list[Callable] = field(default_factory=list)

    # Define post-initialization method to set defaults and validate data
    def __post_init__(self) -> None:
        """
        Post-initialization to set default values and validate data.
        """
        
        # If eval_batch_size is not provided, set it to train_batch_size
        if self.eval_batch_size is None:
            self.eval_batch_size = self.train_batch_size
        
        if not isinstance(self.data_loader.train_data.input, dict) or len(self.data_loader.train_data.input) == 0:
            raise ValueError("train_data.input must be a non-empty dictionary of Tensors")
        
        # Validate that all values in train_data.input are Tensors
        for key, value in self.data_loader.train_data.input.items():
            if not isinstance(value, Tensor):
                raise ValueError(f"All values in train_data.input must be Tensors. Got {type(value)} for key '{key}'")
        
        # Validate valid_data if provided
        if self.data_loader.valid_data is not None:
            if not isinstance(self.data_loader.valid_data, LabeledData):
                raise ValueError("valid_data must be a LabeledData instance")
            
            if not isinstance(self.data_loader.valid_data.input, dict) or len(self.data_loader.valid_data.input) == 0:
                raise ValueError("valid_data.input must be a non-empty dictionary of Tensors")
            
            # Validate that all values in valid_data.input are Tensors
            for key, value in self.data_loader.valid_data.input.items():
                if not isinstance(value, Tensor):
                    raise ValueError(f"All values in valid_data.input must be Tensors. Got {type(value)} for key '{key}'")
            
            # Ensure that train_data and valid_data have the same input keys
            if set(self.data_loader.valid_data.input.keys()) != set(self.data_loader.train_data.input.keys()):
                raise ValueError(
                    f"Validation data input keys {list(self.data_loader.valid_data.input.keys())} "
                    f"must match training data input keys {list(self.data_loader.train_data.input.keys())}"
                )