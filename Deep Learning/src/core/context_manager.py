from typing import Any

# Global flag to disable gradient computation
_NO_GRAD = False

class no_grad:
    
    ### Magic methods ###
    
    def __enter__(self) -> 'no_grad':
        """
        Method to disable gradient computation within the no_grad context
        """
        
        # Import global flag
        global _NO_GRAD
        
        # Store the previous value of the flag
        self.prev = _NO_GRAD
        
        # Disable gradient computation
        _NO_GRAD = True
        
        # Return the instance of the class
        return self


    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """
        Method to enable gradient computation when exiting the no_grad context
        
        Parameters:
        - exc_type (Any): Type of the exception
        - exc_value (Any): Value of the exception
        - traceback (Any): Traceback of the exception
        """
        
        # Import global flag
        global _NO_GRAD
        
        # Enable gradient computation
        _NO_GRAD = self.prev