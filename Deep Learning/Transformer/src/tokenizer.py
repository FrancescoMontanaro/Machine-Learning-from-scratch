import os
import json
import base64
import regex as re


class Tokenizer:
    
    ### Static attributes ###
    
    # The regex patterns to split the text into chunks.
    # The regex pattern is used to force some types of merges not to happen (e.g., merging a space with a letter) 
    # in order to keep the text readable and improve performance
    regex_patterns = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""")
    

    ### Magic methods ###
    
    def __init__(self) -> None:
        """
        Class constructor
        """
        
        # Initialize the attributes
        self.vocab = None
        self.merges = None
        
        # The size of the vocabulary
        self.vocab_size = None
    
    
    ### Public methods ###
    
    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
        """
        Function to train the tokenizer on a text by learning the vocabulary and the merges
        
        Parameters:
        - text (str): The text to train the tokenizer on
        - vocab_size (int): The size of the vocabulary
        - verbose (bool): Whether to print the progress of the training
        
        Raises:
        - ValueError: If the vocabulary size is less than or equal to 256
        """
        
        # Save the vocabulary size
        self.vocab_size = vocab_size
        
        # Compute the number of new tokens to add to the vocabulary
        num_new_tokens = self.vocab_size - 256 # vocab_size - 256 (8 bits, which is the ones of the utf-8 encoding)
        
        # Check if the number of new tokens is valid
        if num_new_tokens <= 0:
            raise ValueError("The vocabulary size must be greater than 256")
        
        # Convert the text into a list of tokens
        tokens = text.encode("utf-8") # Getting the utf-8 encoding of the text (raw bytes)
        tokens = list(map(int, tokens)) # Convert the bytes into a list of integers (0-255)
        
        # Initialize the merges and the vocabulary if they are not initialized
        if self.merges is None: self.merges = {}
        if self.vocab is None: self.vocab = {idx: bytes([idx]) for idx in range(256)}
        
        # Iterate over the number of new tokens to add to the vocabulary
        for idx in range(num_new_tokens):
            # Compute the statistics for the current tokens list
            stats = self._get_stats(list(tokens))
            
            # Get the most common bigram
            top_bigram, _ = max(stats.items(), key=lambda x: x[1])
            
            # Compute the new token by incrementing the current vocab size which starts at 256
            # (the first 256 tokens are the same as the original tokens, i.e., the utf-8 encoding)
            new_token = 256 + idx
            
            # Merge the most common bigram into the new token
            tokens = self._merge(tokens, top_bigram, new_token)
            
            # Store the merge operation
            self.merges[top_bigram] = new_token
            
            # Print the progress if verbose is True
            if verbose:
                print(f"{idx+1}/{num_new_tokens} --> Merged tokens {top_bigram} into token {new_token}")
        
        # Update the vocabulary by adding the original tokens
        for bigram, new_token in self.merges.items():
            self.vocab[new_token] = self.vocab[bigram[0]] + self.vocab[bigram[1]]
            
            
    def encode(self, text: str) -> list[int]:
        """
        Function to encode a text into a list of tokens
        
        Parameters:
        - text (str): The text to encode
        
        Returns:
        - list[int]: A list of tokens
        
        Raises:
        - ValueError: If the tokenizer is not trained
        """
        
        # Check if the tokenizer is trained
        if not isinstance(self.vocab, dict) or not isinstance(self.merges, dict):
            raise ValueError("The tokenizer is not trained. Train the tokenizer before encoding a text or load the parameters from a file")
        
        # Split the text into chunks and filter out the empty chunks
        chunks = self._split_regex(text)
        chunks = [chunk for chunk in chunks if len(chunk) > 0]
        
        # Initialize the tokens list
        tokens = []
        
        # Iterate over the chunks and encode them into tokens
        for chunk in chunks:
            # Initialize the tokens list
            chunk_tokens = list(chunk.encode("utf-8"))
            
            # Loop only if there are at least 2 tokens that can be merged
            while len(chunk_tokens) >= 2:
                # Getting the statistics of the current tokens
                stats = self._get_stats(chunk_tokens)
                
                # Find the pair with the lowest index in the merges dictionary
                # This is because we want to merge the most recent pairs first
                bigram = min(stats, key=lambda x: self.merges.get(x, float("inf"))) # type: ignore
                
                # Check if the pair is in the merges dictionary
                if bigram not in self.merges:
                    break # If the pair is not in the merges dictionary, we can't compress anymore
                
                # Merging the pair into the new token and updating the tokens
                chunk_tokens = self._merge(chunk_tokens, bigram, self.merges[bigram])
        
            # Add the tokens to the list
            tokens.extend(chunk_tokens)

        return tokens
    
    
    def decode(self, tokens: list[int]) -> str:
        """
        Function to decode a list of tokens into a text
        
        Parameters:
        - tokens (list[int]): The list of tokens to decode
        
        Returns:
        - str: The decoded text
        
        Raises:
        - ValueError: If the tokenizer is not trained
        """
        
        # Check if the tokenizer is trained
        if not isinstance(self.vocab, dict) or not isinstance(self.merges, dict):
            raise ValueError("The tokenizer is not trained. Train the tokenizer before decoding a list of tokens or load the parameters from a file")
        
        # Merge the tokens into a sequence of bytes
        bytes_tokens = b"".join([self.vocab[token] for token in tokens])
        
        # Decode the bytes into a string
        return bytes_tokens.decode("utf-8", errors="replace")
    
    
    def save(self, file_path: str) -> None:
        """
        Function to save the parameters of the tokenizer into a json file
        
        Parameters:
        - file_path (str): The path to save the parameters
        
        Raises:
        - ValueError: If the tokenizer is not trained
        - ValueError: If the file is not a json file
        """
        
        # Check if the tokenizer is trained
        if not isinstance(self.vocab, dict) or not isinstance(self.merges, dict):
            raise ValueError("The tokenizer is not trained. Train the tokenizer before saving the parameters or load the parameters from a file")
        
        # Check if the file is a json file
        if not file_path.endswith(".json"):
            raise ValueError("The file must be a json file")
        
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert bytes in vocab to base64 strings in order to serialize the vocab
        serializable_vocab = {k: base64.b64encode(v).decode('utf-8') for k, v in self.vocab.items()}
        
        # Serialize the merges keys to strings in order to save them
        serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        
        # Save the parameters into a json file
        with open(file_path, "w") as f:
            json.dump({"vocab": serializable_vocab, "merges": serializable_merges}, f, indent=4)
            
            
    def load(self, file_path: str) -> None:
        """
        Function to load the parameters of the tokenizer from a json file
        
        Parameters:
        - file_path (str): The path to load the parameters
        
        Raises:
        - ValueError: If the file is not a json file or the data format is invalid
        - FileNotFoundError: If the file does not exist
        """
        
        # Check if the file is a json file
        if not file_path.endswith(".json"):
            raise ValueError("The file must be a json file")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError("The file does not exist")
        
        # Load the parameters from the json file
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Extract the vocabulary and the merges
        vocab = data.get("vocab")
        merges = data.get("merges")
        
        # The vocab must be a dict of integers to bytes
        if not isinstance(vocab, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in vocab.items()):
            raise ValueError("The vocabulary is invalid. It must be a dictionary of integers to bytes encoded in base64")
        
        # The merges must be a dict of tuples to integers
        if not isinstance(merges, dict) or not all(isinstance(k, str) and isinstance(v, int) for k, v in merges.items()):
            raise ValueError("The merges are invalid. It must be a dictionary of tuples to integers")
        
        # Decode the base64 strings in the vocab
        vocab = {int(k): base64.b64decode(v) for k, v in vocab.items()}
        
        # Deserialize the merges keys
        merges = {tuple(map(int, k.split(","))): v for k, v in merges.items()}
        
        # Update the attributes
        self.vocab = vocab
        self.merges = merges
        self.vocab_size = len(vocab.keys())
        
    
    ### Protected methods ###
    
    @classmethod
    def _split_regex(cls, text: str) -> list[str]:
        """
        Function to split the text into chunks by applying a regex pattern in order to force some types of merges not to happen
        
        Parameters:
        - text (str): The text to be split
        
        Returns:
        - list[str]: A list of strings that are the result of the split
        """
        
        # Split the text into chunks by applying the regex pattern
        return re.findall(cls.regex_patterns, text)
      

    @staticmethod
    def _get_stats(tokens: list[int]) -> dict[tuple[int, int], int]:
        """
        Function to get the statistics from the list of tokens.
        The statistics are the number of times each pair of tokens appears in the list
        
        Parameters:
        - tokens (list[int]): The list of tokens
        
        Returns:
        - dict[tuple[int, int], int]: A dictionary that contains the statistics of the tokens
        """
        
        # Create a dictionary to store the counts of the tokens
        counts = {}
        
        # Iterate over the tokens and count the number of times each pair appears
        for pair in zip(tokens, tokens[1:]):
            # Increment the count of the pair
            counts[pair] = counts.get(pair, 0) + 1
        
        return counts
    
    
    @staticmethod
    def _merge(tokens: list[int], pair: tuple[int, int], new_token: int) -> list[int]:
        """
        Function to merge a pair of tokens into a new token
        
        Parameters:
        - tokens (list[int]): The list of tokens
        - pair (tuple[int, int]): The pair of tokens to be merged
        - new_token (int): The new token to replace the pair
        """
        
        # Create a new list to store the new tokens
        new_tokens = []
        
        # Initialize the index
        i = 0
        
        # Loop until the end of the tokens list
        while i < len(tokens):
            # The current bigram is the same as the pair
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                # Add the new token to the list and increment the index by 2
                new_tokens.append(new_token)
                i += 2
                
            # The current bigram is different from the pair
            else:
                # Add the current token to the list and increment the index by 1
                new_tokens.append(tokens[i])
                i += 1
                
        return new_tokens