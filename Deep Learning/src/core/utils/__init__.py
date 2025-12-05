# Static no-op function to avoid creating new lambda objects
def _noop() -> None:
    """
    Static no-op function used to clear backward references.
    """
    
    pass