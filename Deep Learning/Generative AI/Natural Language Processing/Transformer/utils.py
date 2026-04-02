import os
import numpy as np


def load_txt_file(path: str) -> str:
    """
    Load a text file from the specified path.
    
    Parameters:
    - path (str): The path to the text file.
    
    Returns:
    - str: The contents of the text file.
    """
    
    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f'The file "{path}" does not exist.')
    
    # Read the file
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()


def build_sequences(
    input_data: np.ndarray, 
    seq_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sequences
    
    Parameters:
    - input_data: np.ndarray, input features
    - seq_length: int, length of input sequences
    - Generates all possible next-token sequence pairs from the input stream.
    
    Returns:
    - X: np.ndarray, input sequences of shape (n - seq_length, seq_length)
    - y: np.ndarray, target sequences of shape (n - seq_length, seq_length)
    """

    # Extract the number of tokens in the input data and calculate the number of possible sliding windows.
    n = len(input_data)
    num_windows = n - seq_length
    
    # Check if the number of possible windows is positive
    if num_windows <= 0:
        raise ValueError(f"Input data too short ({n}) for seq_length={seq_length}.")

    # Build all sliding windows deterministically.
    X = np.array(
        [input_data[i : i + seq_length] for i in range(num_windows)],
        dtype = np.int32
    )

    # Build target sequences by shifting the input sequences by one position to the right.
    y = np.array(
        [input_data[i + 1 : i + seq_length + 1] for i in range(num_windows)],
        dtype = np.int32
    )

    # Return the input sequences and target sequences as numpy arrays.
    return X, y
