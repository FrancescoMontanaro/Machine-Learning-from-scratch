import numpy as np
from typing import Union, Tuple, List, Optional

from ..tensor import Tensor
from .constants import EPSILON


def shuffle_data(data: Union[Tensor, Tuple[Tensor, ...]]) -> Tuple[Union[Tensor, Tuple[Tensor, ...]], np.ndarray]:
    """
    Method to shuffle the dataset
    
    Parameters:
    - data (Union[Tensor, Tuple[Tensor, ...]]): Dataset to shuffle. Can be a single tensor or a tuple of tensors.
    
    Returns:
    - Tuple[Union[Tensor, Tuple[Tensor, ...]], np.ndarray]: Shuffled dataset and the indices used for shuffling
    
    Raises:
    - ValueError: If the data is not a tensor or a tuple of tensors
    - ValueError: If tensors in the tuple have different batch sizes
    """
    
    # Check if the data is a tuple of tensors
    if isinstance(data, tuple) and len(data) >= 2:
        # Validate that all tensors have the same batch size
        n_samples = data[0].shape[0]
        for i, tensor in enumerate(data[1:], 1):
            if tensor.shape[0] != n_samples:
                raise ValueError(f"All tensors must have the same batch size. Tensor 0 has {n_samples} samples, tensor {i} has {tensor.shape[0]} samples")

        # Generate random indices
        indices = np.random.permutation(n_samples)
        
        # Shuffle all tensors using the same indices
        shuffled_tensors = tuple(tensor[indices] for tensor in data)
        
        # Return the shuffled dataset
        return shuffled_tensors, indices
    
    # Check if the data is a single tensor
    elif isinstance(data, Tensor):
        # Get the number of samples
        n_samples = data.shape[0]
        
        # Generate random indices
        indices = np.random.permutation(n_samples)
        
        # Shuffle the dataset
        data_shuffled = data[indices]
        
        # Return the shuffled dataset
        return data_shuffled, indices
    
    else:
        # Raise a ValueError if the data is not a tensor or a tuple of tensors
        raise ValueError("data must be a tensor or a tuple of tensors")


def split_data(data: Union[Tensor, Tuple[Tensor, ...]], split_pct: float = 0.1, shuffle: bool = False) -> Tuple[Tuple[Tensor, ...], Optional[np.ndarray]]:
    """
    Splits datasets into two subsets: for example, training and testing sets.
    The input dataset can be a single tensor or a tuple of tensors (X, y).
    
    Parameters:
    - data (Union[Tensor, Tuple[Tensor, Tensor]]): Dataset
    - split_pct (float): Split percentage for the test set.
    - shuffle (bool): Whether to shuffle the dataset before splitting.
    
    Returns:
    - Tuple[Tuple[Tensor, ...], Optional[np.ndarray]]: Tuple containing the training and testing sets, and optionally the indices used for shuffling.
    
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
        
        # Initialize indices to None
        indices = None
        
        # Check if the data should be shuffled
        if shuffle:
            # Shuffle the data
            (X, y), indices = shuffle_data((X, y))
            
        # Get the number of samples
        n_samples = X.shape[0]
        
        # Compute the test size
        test_size = max(1, int(n_samples * split_pct))
        
        # Split the data
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        # Return the split data
        return (X_train, X_test, y_train, y_test), indices
    
    # Check if the data is a tensor
    elif isinstance(data, Tensor):
        # Initialize indices to None
        indices = None
        
        # Check if the data should be shuffled
        if shuffle:
            # Shuffle the data
            data, indices = shuffle_data(data)
            
        if not isinstance(data, Tensor):
            # Raise a ValueError if the data is not a tensor or a tuple of tensors
            raise ValueError("data must be a tensor or a tuple of two tensors")
            
        # Get the number of samples
        n_samples = data.shape[0]
        
        # Compute the test size
        test_size = max(1, int(n_samples * split_pct))
        
        # Split the data
        X_train, X_test = data[:-test_size], data[-test_size:]
        
        # Return the split data
        return (X_train, X_test), indices
        
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
    one_hot = np.zeros((y.shape[0], n_classes))
    
    # Set the appropriate index to 1
    one_hot[np.arange(y.shape[0]), y.data.flatten().astype(int)] = 1
    
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


def split(tensor: 'Tensor', indices_or_sections: Union[int, List[int]], axis: int = 0) -> List['Tensor']:
    """
    Method to split the tensor into multiple sub-tensors
    
    Parameters:
    - tensor (Tensor): Tensor to split
    - indices_or_sections (Union[int, List[int]]): Number of sections or list of indices to split the tensor
    - axis (int): Axis along which to split the tensor
    
    Returns:
    - List[Tensor]: List of sub-tensors
    """
    
    # Compute and return the list of sub-tensors
    return Tensor.split(tensor, indices_or_sections, axis)


def einsum(subscripts: str, *operands: 'Tensor') -> 'Tensor':
    """
    Method to perform the einsum operation on the given operands
    
    Parameters:
    - subscripts (str): Einsum subscript string
    - operands (Tensor): Input tensors
    
    Returns:
    - Tensor: Result of the einsum operation
    """
    
    # Compute and return the result of the einsum operation
    return Tensor.einsum(subscripts, *operands)