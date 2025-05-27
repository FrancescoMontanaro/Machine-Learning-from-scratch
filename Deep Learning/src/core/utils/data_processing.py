import numpy as np
from typing import Union, Tuple, List, Optional

from ..tensor import Tensor
from .constants import EPSILON


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


def split_data(data: Union[Tensor, Tuple[Tensor, Tensor]], split_pct: float = 0.1, shuffle: bool = False) -> Tuple[Tensor, ...]:
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


def compute_stats(X: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute normalization statistics from training data only.
    
    Parameters:
    - X: Tensor, training data
    - axis: Optional[Union[int, Tuple[int, ...]]], axis along which to compute the statistics
    
    Returns:
    - tuple[Tensor, Tensor, Tensor, Tensor], mean, std, min, max
    """
    
    # Extract the data from the Tensor
    data = X.to_numpy()
    
    # Compute mean, std, min, and max across the first two axes (time and features)
    min = np.min(data, axis=axis)
    max = np.max(data, axis=axis)
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    
    # Ensure std is not zero to avoid division by zero
    std = np.maximum(std, EPSILON)
    
    # Return as Tensors
    return Tensor(mean), Tensor(std), Tensor(min), Tensor(max)


def z_score_normalize(X: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """
    Z-score normalization
    
    Parameters:
    - X: Tensor, input data to normalize
    - mean: Tensor, mean for normalization
    - std: Tensor, standard deviation for normalization
    
    Returns:
    - Tensor, normalized data
    """
    
    # Compute Z-score normalization
    return (X - mean) / std


def z_score_denormalize(X: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """
    Z-score denormalization
    
    Parameters:
    - X: Tensor, normalized data to denormalize
    - mean: Tensor, mean used for normalization
    - std: Tensor, standard deviation used for normalization
    
    Returns:
    - Tensor, denormalized data
    """
    
    # Compute denormalization
    return X * std + mean


def min_max_normalize(X: Tensor, min: Tensor, max: Tensor) -> Tensor:
    """
    Min-Max normalization
    
    Parameters:
    - X: Tensor, input data to normalize
    - min: Tensor, minimum for normalization
    - max: Tensor, maximum for normalization
    
    Returns:
    - Tensor, normalized data
    """
    
    # Compute Min-Max normalization
    return (X - min) / (max - min + EPSILON)


def min_max_denormalize(X: Tensor, min: Tensor, max: Tensor) -> Tensor:
    """
    Min-Max denormalization
    
    Parameters:
    - X: Tensor, normalized data to denormalize
    - min: Tensor, minimum used for normalization
    - max: Tensor, maximum used for normalization
    
    Returns:
    - Tensor, denormalized data
    """
    
    # Compute denormalization
    return X * (max - min) + min


def concat(tensors: List['Tensor'], axis: int = 0) -> 'Tensor':
    """
    Method to concatenate the tensors along the specified axis
    
    Parameters:
    - tensors (Tensor): List of tensors to concatenate
    - axis (int): Axis along which to concatenate the tensors
    
    Returns:
    - Tensor: Concatenated tensor
    """
    
    # Compute and return the concatenated tensor
    return Tensor.concat(tensors, axis)


def stack(tensors: List['Tensor'], axis: int = 0) -> 'Tensor':
    """
    Method to stack the tensors along the specified axis
    
    Parameters:
    - tensors (Tensor): List of tensors to stack
    - axis (int): Axis along which to stack the tensors
    
    Returns:
    - Tensor: Stacked tensor
    """
    
    # Compute and return the stacked tensor
    return Tensor.stack(tensors, axis)