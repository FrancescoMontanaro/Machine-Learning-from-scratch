from typing import Any, Tuple, Optional

# Global buffer for saving data to be exchanged between forward and backward passes.
_DATA_TAPE: list[Tuple[Any, ...] | None] = []


def tape_push(data: Tuple[Any, ...]) -> int:
    """
    Save data from forward and return an index.
    
    Parameters:
    - data (Tuple[Any, ...]): Data to be saved.
    """
    
    # Extract the index for the new data
    idx = len(_DATA_TAPE)
    
    # Add the arguments to the data tape
    _DATA_TAPE.append(data)
    
    # Return the index of the saved data
    return idx


def tape_pop(idx: int) -> Optional[Tuple[Any, ...]]:
    """
    Retrieve the saved tuple and free the reference.
    
    Parameters:
    - idx (int): Index of the saved data.
    
    Returns:
    - Tuple[Any, ...]: The saved data tuple, or None if the index is invalid.
    """
    
    # Check if the index is valid
    if idx < 0:
        # If the index is invalid, return None
        return None
    
    # Extract the data from the tape
    data = _DATA_TAPE[idx]
    
    # Remove the reference to the saved data to avoid memory leaks
    _DATA_TAPE[idx] = None
    
    # Return the saved data
    return data