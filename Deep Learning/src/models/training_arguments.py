from dataclasses import dataclass, field
from typing import Optional, Callable, Dict

from ..core import Tensor
from ..optimizers import Optimizer
from ..loss_functions import LossFn


@dataclass
class LabeledData:
    """
    Container for input data and corresponding target labels.
    
    Parameters:
    - input (Dict[str, Tensor]): Dictionary of named input tensors
    - target (Tensor): Target tensor (labels)
    """
    
    input: Dict[str, Tensor]
    target: Tensor


@dataclass
class TrainingArguments:
    """
    Configuration for training parameters.
    
    Parameters:
    - train_data (LabeledData): Training data containing inputs and targets
    - optimizer (Optimizer): Optimizer to use for training
    - loss_fn (LossFn): Loss function to use for training
    - valid_data (Optional[LabeledData]): Validation data containing inputs and targets (default: None)
    - train_batch_size (int): Number of samples per training batch (default: 32)
    - eval_batch_size (Optional[int]): Number of samples per evaluation batch (default: None, uses train_batch_size)
    - num_epochs (int): Number of epochs to train the model (default: 10)
    - shuffle (bool): Whether to shuffle the training data between epochs (default: True)
    - gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating the parameters (default: 1)
    - metrics (list[Callable[..., Tensor]]): List of metrics to compute during training (default: [])
    - callbacks (list[Callable]): List of callbacks to execute during training (default: [])
    """
    
    # Mandatory parameters - Data
    train_data: LabeledData
    
    # Mandatory parameters - Training
    optimizer: Optimizer
    loss_fn: LossFn
    
    # Optional parameters - Validation data
    valid_data: Optional[LabeledData] = None
    
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
        
        # Validate train_data
        if not isinstance(self.train_data, LabeledData):
            raise ValueError("train_data must be a LabeledData instance")
        
        if not isinstance(self.train_data.input, dict) or len(self.train_data.input) == 0:
            raise ValueError("train_data.input must be a non-empty dictionary of Tensors")
        
        # Validate that all values in train_data.input are Tensors
        for key, value in self.train_data.input.items():
            if not isinstance(value, Tensor):
                raise ValueError(f"All values in train_data.input must be Tensors. Got {type(value)} for key '{key}'")
        
        # Validate valid_data if provided
        if self.valid_data is not None:
            if not isinstance(self.valid_data, LabeledData):
                raise ValueError("valid_data must be a LabeledData instance")
            
            if not isinstance(self.valid_data.input, dict) or len(self.valid_data.input) == 0:
                raise ValueError("valid_data.input must be a non-empty dictionary of Tensors")
            
            # Validate that all values in valid_data.input are Tensors
            for key, value in self.valid_data.input.items():
                if not isinstance(value, Tensor):
                    raise ValueError(f"All values in valid_data.input must be Tensors. Got {type(value)} for key '{key}'")
            
            # Ensure that train_data and valid_data have the same input keys
            if set(self.valid_data.input.keys()) != set(self.train_data.input.keys()):
                raise ValueError(
                    f"Validation data input keys {list(self.valid_data.input.keys())} "
                    f"must match training data input keys {list(self.train_data.input.keys())}"
                )