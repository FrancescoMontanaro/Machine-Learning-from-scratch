import math
from enum import Enum
from typing import Optional, Tuple, Generator

from ..core import Tensor
from .training_arguments import LabeledData
from ..core.utils.data_processing import shuffle_data


class Split(Enum):
    """
    Enumeration for data splits.
    """

    TRAIN = "train"
    VALID = "valid"


class DataLoader:

    #####################
    ### Magic methods ###
    #####################

    def __init__(
        self, 
        train_data: LabeledData, 
        valid_data: Optional[LabeledData] = None
    ) -> None:
        """
        Constructor for the DataLoader class.
        
        Args:
            train_data (LabeledData): The training data.
            valid_data (Optional[LabeledData]): The validation data. Defaults to None.
        """

        # Initialize the DataLoader with the provided training, validation, and test data.
        self.train_data = train_data
        self.valid_data = valid_data

        # Cache batch counts to avoid recalculation
        self._batch_counts = {}


    ######################
    ### Public methods ###
    ######################

    def get_batch(self, split: Split, batch_size: int) -> Generator[Tuple[Tuple[Tensor, ...], Tensor], None, None]:
        """
        Get a batch of data from the training set.
        
        Args:
            split (Split): The split to retrieve the batch from. Can be either Split.TRAIN or Split.VAL.
            batch_size (int): The size of the batch to retrieve.

        Returns:
            Generator[Tuple[int, Tuple[Tuple[Tensor, ...], Tensor]], None, None]: A generator that yields batches of data from the specified split.
        """

        # Select the appropriate data based on the specified split.
        data = self._get_data_split(split)

        # Extract the input data from the LabeledData object.
        data_inputs = data.input_tuple
        data_targets = data.target

        # Calculate the number of batches for the specified split and batch size.
        num_steps = self.num_batches(split, batch_size)

        # Yield batches
        for step in range(num_steps):
            # Slice the input data and target data to get the batch for the specified step and batch size.
            x_batch = tuple(tensor[step * batch_size:(step + 1) * batch_size] for tensor in data_inputs)
            y_batch = data_targets[step * batch_size:(step + 1) * batch_size]

            # Yield the batch of data as a tuple of input tensors and a target tensor.
            yield x_batch, y_batch

    
    def num_batches(self, split: Split, batch_size: int) -> int:
        """
        Get the number of batches for the specified split and batch size.
        
        Args:
            split (Split): The split to calculate the number of batches for. Can be either Split.TRAIN or Split.VAL.
            batch_size (int): The size of the batches.
        
        Returns:
            int: The number of batches for the specified split and batch size.
        """

        # Select the appropriate data based on the specified split.
        data = self._get_data_split(split)

        # Extract the input data from the LabeledData object.
        data_inputs = data.input_tuple

        # Calculate the number of batches (cached per split and batch_size)
        cache_key = (split, batch_size)
        if cache_key not in self._batch_counts:
            self._batch_counts[cache_key] = max(1, math.ceil(data_inputs[0].shape[0] / batch_size))

        # Get the number of batches for the specified split and batch size from the cache.
        num_steps = self._batch_counts[cache_key]

        return num_steps
    

    def shuffle(self, split: Split) -> LabeledData:
        """
        Shuffle the data in the specified split.
        
        Args:
            split (Split): The split to shuffle. Can be either Split.TRAIN or Split.VAL.

        """

        # Select the appropriate data based on the specified split.
        data = self._get_data_split(split)

        # Extract the input data and target data from the LabeledData object.
        data_inputs = data.input_tuple
        y_data = data.target

        # Shuffle the data
        shuffled_data, _ = shuffle_data((*data_inputs, y_data))
        
        # Cast since we know shuffle_data returns a tuple when given a tuple
        assert isinstance(shuffled_data, tuple), "shuffle_data should return a tuple"

        # Extract shuffled inputs and targets
        inputs_shuffled: Tuple[Tensor, ...] = shuffled_data[:-1] 
        y_shuffled: Tensor = shuffled_data[-1]

        # Update the LabeledData object with the shuffled data.
        data.input = {key: inputs_shuffled[i] for i, key in enumerate(data.input.keys())}
        data.target = y_shuffled

        # Return the shuffled data.
        return data
    

    #########################
    ### Protected methods ###
    #########################

    def _get_data_split(self, split: Split) -> LabeledData:
        """
        Get the data for the specified split.
        
        Args:
            split (Split): The split to retrieve the data from. Can be either Split.TRAIN or Split.VAL.

        Returns:
            LabeledData: The data for the specified split.
        """

        # Select and return the appropriate data based on the specified split.
        if split == Split.TRAIN:
            return self.train_data
        elif split == Split.VALID:
            # Check if validation data is provided before attempting to access it.
            if self.valid_data is None:
                raise ValueError("Validation data is not provided.")
            return self.valid_data
        else:
            # If an invalid split is specified, raise a ValueError.
            raise ValueError("Invalid split. Must be Split.TRAIN, Split.VALID, or Split.TEST.")