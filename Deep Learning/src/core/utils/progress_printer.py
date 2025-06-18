import shutil


class ProgressPrinter:
    
    ### Magic Methods ###
    
    def __init__(self):
        """
        Initialize the ProgressPrinter with a last line length tracker.
        """
        
        self.last_line_length = 0
        
        
    ### Public Methods ###
        
    def print_progress(self, message: str, end: str = "", flush: bool = True):
        """
        Print progress message, clearing previous line if needed.
        
        Parameters:
        - message (str): The message to print.
        - end (str): The string appended after the message. Defaults to "".
        - flush (bool): Whether to forcibly flush the output buffer. Defaults to True.
        """
        
        # Clear previous line if it was longer
        if len(message) < self.last_line_length:
            # Clear the remaining characters from the previous message
            clear_chars = " " * (self.last_line_length - len(message))
            
            # Print the message with cleared characters
            print(f"\r{message}{clear_chars}", end=end, flush=flush)
            
        # Print the new message
        else:
            # Print the message directly
            print(f"\r{message}", end=end, flush=flush)
        
        # Update the last line length
        self.last_line_length = len(message)
    
    
    def print_final(self, message: str):
        """
        Print final message and reset.
        
        Parameters:
        - message (str): The final message to print.
        """
        
        # Get terminal width
        terminal_width = shutil.get_terminal_size().columns
        
        # Clear the last printed line
        print(f"\r{' ' * (terminal_width-1)}", end="")
        
        # Print the final message
        print(f"\r{message}")
        
        # Reset the last line length tracker
        self.last_line_length = 0