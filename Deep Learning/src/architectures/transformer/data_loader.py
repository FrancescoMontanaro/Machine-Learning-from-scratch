import numpy as np
from typing import Literal

from ...core import Tensor


class DataLoader:
    
    ### Magic methods ###
    
    def __init__(self, data: Tensor, train_val_split: float) -> None:
        """
        Initialize the data loader.
        
        Parameters:
        - data (Tensor): The data to load.
        - train_val_split (float): The proportion of the data to use for training.
        """
        
        # Store the data
        self.data = data
        
        # Compute the split index
        n = int(train_val_split * len(data.data))

        # Split the data into training and validation sets
        self.train_data = data[:n]
        self.val_data = data[n:]
    
    
    ### Public methods ###
    
    def get_batch(self, split: Literal["train", "validation"] = "validation", batch_size: int = 4, sequence_length: int = 8) -> tuple[Tensor, Tensor]:
        """
        Method that returns a batch of data.
        
        Parameters:
        - split (str): The split to use for the data (either "train" or "validation").
        - batch_size (int): The size of the batch.
        - sequence_length (int): The size of the block (sequence length).
        
        Returns:
        - tuple[Tensor, Tensor]: The input and target sequences.
        """
        
        # Select the data based on the split
        data = self.train_data if split == "train" else self.val_data
        
        # Randomly select n starting indices for the sequences (n=batch_size)
        ix = np.random.randint(0, len(data.data) - sequence_length, (batch_size,))
        
        # Create the input and target sequences by selecting a block of text from the data
        # The target sequence is the input sequence shifted by one character
        x = Tensor(np.stack([data.data[i:i+sequence_length] for i in ix]), dtype=np.int32)
        y = Tensor(np.stack([data.data[i+1:i+sequence_length+1] for i in ix]), dtype=np.int32)
        
        # Return the input and target sequences
        return x, y