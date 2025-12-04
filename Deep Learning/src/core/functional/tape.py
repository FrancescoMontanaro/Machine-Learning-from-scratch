from typing import Any, Tuple, Optional

# Global buffer for saving data to be exchanged between forward and backward passes.
# Using a dictionary instead of a list allows true deletion of entries, preventing memory growth.
_DATA_TAPE: dict[int, Tuple[Any, ...]] = {}

# Counter for generating unique keys (monotonically increasing)
_TAPE_COUNTER: int = 0


def tape_push(data: Tuple[Any, ...]) -> int:
    """
    Save data from forward and return a unique key.
    
    Parameters:
    - data (Tuple[Any, ...]): Data to be saved.
    
    Returns:
    - int: Unique key to retrieve the data later.
    """
    
    # Use the global counter and data tape
    global _TAPE_COUNTER
    
    # Get the current counter value as the key
    key = _TAPE_COUNTER
    
    # Increment the counter for the next push
    _TAPE_COUNTER += 1
    
    # Store the data in the dictionary
    _DATA_TAPE[key] = data
    
    # Return the key
    return key


def tape_pop(key: int) -> Optional[Tuple[Any, ...]]:
    """
    Retrieve the saved tuple and remove it from the tape (true deletion).
    
    Parameters:
    - key (int): Key of the saved data.
    
    Returns:
    - Tuple[Any, ...]: The saved data tuple, or None if the key is invalid.
    """
    
    # Check if the key is valid (negative keys are used to indicate no saved data)
    if key < 0:
        return None
    
    # Pop the data from the dictionary (returns None if key doesn't exist)
    # This truly removes the entry, freeing memory
    return _DATA_TAPE.pop(key, None)


def tape_clear() -> None:
    """
    Clear all entries from the tape and reset the counter.
    Useful for explicit cleanup at the end of training epochs or batches.
    """

    # Use the global counter and data tape
    global _TAPE_COUNTER
    
    # Clear all entries
    _DATA_TAPE.clear()
    
    # Reset counter to prevent integer overflow in extremely long training sessions
    _TAPE_COUNTER = 0


def tape_size() -> int:
    """
    Get the current number of entries in the tape.
    Useful for debugging memory issues.
    
    Returns:
    - int: Number of entries currently in the tape.
    """
    
    # Return the size of the data tape
    return len(_DATA_TAPE)