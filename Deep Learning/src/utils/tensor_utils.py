import numpy as np
from typing import Optional, Union, Tuple

from ..core import Tensor


def shuffle_data(data: Union[Tensor, Tuple[Tensor, Tensor]]) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Method to shuffle the dataset
    
    Parameters:
    - data (Union[Tensor, Tuple[Tensor, Tensor]]): Dataset to shuffle
    
    Returns:
    - Union[Tensor, Tuple[Tensor, Tensor]]: Shuffled dataset
    
    Raises:
    - ValueError: If the data is not a tensor or a tuple
    """
    
    # Check if the data is a tensor or a tuple of tensors
    if isinstance(data, tuple) and len(data) == 2:
        # Unpack the data
        X, y = data
    
        # Get the number of samples
        n_samples = X.shape()[0]
        
        # Generate random indices
        indices = np.random.permutation(n_samples)
        
        # Shuffle the dataset
        X_shuffled = X[indices]
        y_shuffled = y[indices]
    
        # Return the shuffled dataset
        return X_shuffled, y_shuffled
    
    # Check if the data is a tensor
    elif isinstance(data, Tensor):
        # Get the number of samples
        n_samples = data.shape()[0]
        
        # Generate random indices
        indices = np.random.permutation(n_samples)
        
        # Shuffle the dataset
        data_shuffled = data[indices]
        
        # Return the shuffled dataset
        return data_shuffled
    
    else:
        # Raise a ValueError if the data is not a tensor or a tuple of tensors
        raise ValueError("data must be a tensor or a tuple of two tensors")


def split_data(data: Union[Tensor, Tuple[Tensor, Tensor]], split_pct: float = 0.1, shuffle: bool = False) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """
    Splits datasets into two subsets: for example, training and testing sets.
    The input dataset can be a single tensor or a tuple of tensors (X, y).
    
    Parameters:
    - data (Union[Tensor, Tuple[Tensor, Tensor]]): Dataset
    - split_pct (float): Split percentage for the test set.
    - shuffle (bool): Whether to shuffle the dataset before splitting.
    
    Returns:
    - Union[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]: Dataset split into training and testing sets.
    
    Raises:
    - ValueError: If test_size is not between 0 and 1.
    - ValueError: If the data is not a tensor or a tuple of tensors.
    """
    
    # Check if the test_size is between 0 and 1
    if not (0.0 < split_pct < 1.0):
        # Raise a ValueError if the test_size is not between 0 and 1
        raise ValueError("test_size must be a float between 0 and 1")
    
    # Set the random seed if provided
    if isinstance(data, tuple) and len(data) == 2:
        # Unpack the data
        X, y = data
        
        # Check if the data should be shuffled
        if shuffle:
            # Shuffle the data
            X, y = shuffle_data((X, y))
            
        # Get the number of samples
        n_samples = X.shape()[0]
        
        # Compute the test size
        test_size = max(1, int(n_samples * split_pct))
        
        # Split the data
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        # Return the split data
        return X_train, X_test, y_train, y_test
    
    # Check if the data is a tensor
    elif isinstance(data, Tensor):
        # Check if the data should be shuffled
        if shuffle:
            # Shuffle the data
            data = shuffle_data(data)
            
        if not isinstance(data, Tensor):
            # Raise a ValueError if the data is not a tensor or a tuple of tensors
            raise ValueError("data must be a tensor or a tuple of two tensors")
            
        # Get the number of samples
        n_samples = data.shape()[0]
        
        # Compute the test size
        test_size = max(1, int(n_samples * split_pct))
        
        # Split the data
        X_train, X_test = data[:-test_size], data[-test_size:]
        
        # Return the split data
        return X_train, X_test
        
    else:
        # Raise a ValueError if the data is not a tensor or a tuple of tensors
        raise ValueError("data must be a tensor or a tuple of two tensors")


def one_hot_encoding(y: Tensor, n_classes: int) -> Tensor:
    """
    Method to perform one-hot encoding on the target variable
    
    Parameters:
    - y (Tensor): Target variable
    - n_classes (int): Number of classes
    
    Returns:
    - Tensor: One-hot encoded target variable
    """
    
    # Initialize the one-hot encoded target variable
    one_hot = np.zeros((y.shape()[0], n_classes))
    
    # Set the appropriate index to 1
    one_hot[np.arange(y.shape()[0]), y.data.astype(int)] = 1
    
    # Return the one-hot encoded target variable
    return Tensor(one_hot, requires_grad=False, dtype=np.int8)