import csv
import numpy as np
from datetime import datetime


def load_data(file_path: str) -> list[dict]:
    """
    Load data from a CSV file with proper date parsing.
    
    Parameters:
    - file_path: str, path to the CSV file
    
    Returns:
    - data: list[dict], list of dictionaries containing the data
    """
    
    # Create a list to hold the data
    data = []
    
    # Open the CSV file and read its contents
    with open(file_path, 'r') as file:
        # Use DictReader to read the CSV file into a list of dictionaries
        reader = csv.DictReader(file)
        
        # Iterate over each row in the CSV file
        for row in reader:
            # Ensure the 'Date' field is in the correct format
            try:
                # Parse the date and convert it to a datetime object
                datetime.strptime(row['Date'], '%Y-%m-%d')
                
                # Add the row to the data list
                data.append(row)
                
            except ValueError:
                # If the date is invalid, skip this row
                print(f"Skipping row with invalid date: {row['Date']}")
                continue
    
    # Sort the data by date
    data.sort(key=lambda x: x['Date'])
    
    # return the loaded and sorted data
    return data


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute log returns: ln(P_t / P_t-1).

    Args:
    - prices (np.ndarray): Array of price values.

    Returns:
    - np.ndarray: Array of log returns.
    """

    # Compute log returns using numpy for efficient array operations
    return np.log(prices[1:] / prices[:-1])


def compute_realized_volatility(log_returns: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling realized volatility.

    Args:
    - log_returns (np.ndarray): Array of log returns.
    - window (int): The rolling window size for volatility calculation.

    Returns:
    - np.ndarray: Array of realized volatility values.
    """

    # Compute squared log returns
    squared_returns = log_returns ** 2
    volatility = []
    for i in range(window, len(squared_returns)):
        volatility.append(np.sqrt(np.mean(squared_returns[i-window:i], axis=0)))

    # Return the computed volatility as a numpy array
    return np.array(volatility, dtype=np.float32)


def directional_accuracy(pred: np.ndarray, true: np.ndarray, close_price_idx: int) -> float:
    """
    Compute directional accuracy on the close-price feature.

    Args:
    - pred (np.ndarray): Predicted values, shape (n_samples, n_features).
    - true (np.ndarray): True values, shape (n_samples, n_features).
    - close_price_idx (int): The index of the close price feature in the arrays.

    Returns:
    - float: The directional accuracy as a percentage.
    """

    # Extract the close price feature from predictions and true values
    pred_np = pred[:, close_price_idx]
    true_np = true[:, close_price_idx]

    # Compute the direction of price movement for predictions and true values
    pred_dir = (pred_np[1:] > pred_np[:-1]).astype(np.int8)
    true_dir = (true_np[1:] > true_np[:-1]).astype(np.int8)

    # Compute and return the directional accuracy
    return float(np.mean(pred_dir == true_dir)) if len(pred_dir) > 0 else 0.0


def build_sequences(input_data: np.ndarray, target_data: np.ndarray, timestamps: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sequences
    
    Parameters:
    - input_data: np.ndarray, input features
    - target_data: np.ndarray, target values (aligned with input_data)
    - timestamps: np.ndarray, timestamps corresponding to input_data
    - seq_length: int, length of input sequences
    
    Returns:
    - tuple[np.ndarray, np.ndarray, np.ndarray]: input sequences, target values, and target timestamps as numpy arrays
    """
    
    # Initialize lists to hold sequences, targets, and target timestamps
    X, y, ts = [], [], []
    
    # Iterate over the input data to create sequences
    for i in range(seq_length, len(input_data)):
        # Append the sequence of input and the corresponding target
        X.append(input_data[i-seq_length:i]) # Sequence of input features
        y.append(target_data[i]) # Target is the next value after the sequence
        ts.append(timestamps[i]) # Timestamp corresponding to the target value
    
    # Convert the lists to numpy arrays
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(ts, dtype=np.int32)
    )
