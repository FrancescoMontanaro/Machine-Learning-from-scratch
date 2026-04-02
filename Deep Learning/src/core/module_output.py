from typing import Iterator

from .tensor import Tensor


class ModuleOutput:
    """
    Standardized output container for all modules.
    
    Wraps a primary tensor (`output`) that flows through the module chain,
    plus optional named auxiliary tensors (e.g., `mu`, `logvar` for VAE).
    """
    
    #####################
    ### Magic methods ###
    #####################
    
    def __init__(self, output: Tensor, **aux: Tensor) -> None:
        """
        Initialize the module output.
        
        Parameters:
        - output (Tensor): The primary output tensor
        - **aux (Tensor): Named auxiliary tensors (e.g., mu=..., logvar=...)
        """
        
        # Store the primary tensor and auxiliary tensors
        self.output = output
        self.aux = aux


    def __repr__(self) -> str:
        """
        String representation of the module output.
        
        Returns:
        - str: A string representation of the module output
        """
        
        # Format auxiliary tensor names and shapes
        aux_str = ", ".join(f"{k}={v.shape}" for k, v in self.aux.items())
        
        # Return the string representation
        return f"ModuleOutput(output={self.output.shape}" + (f", {aux_str})" if aux_str else ")")
    
    
    def __getattr__(self, name: str) -> Tensor:
        """
        Access auxiliary tensors as attributes.
        
        Parameters:
        - name (str): Name of the auxiliary tensor
        
        Returns:
        - Tensor: The auxiliary tensor
        
        Raises:
        - AttributeError: If the auxiliary tensor is not found
        """
        
        # Check if the attribute is an auxiliary tensor
        if name in ("output", "aux"):
            raise AttributeError(name)
            
        # Try to find the attribute in the auxiliary tensors
        aux = object.__getattribute__(self, "aux")
        if name in aux:
            return aux[name]
        
        # Raise an error if the attribute is not found
        raise AttributeError(f"No auxiliary tensor '{name}' in ModuleOutput. Available: {list(aux.keys())}")


    def __iter__(self) -> Iterator[Tensor]:
        """
        Iterate over the primary tensor and auxiliary tensors.
        Yields the primary tensor first, then auxiliary tensors in insertion order.
        
        Returns:
        - Iterator[Tensor]: An iterator over the tensors
        """
        
        # Yield the primary tensor
        yield self.output
        
        # Yield the auxiliary tensors
        yield from self.aux.values()
    

    ######################
    ##### Properties #####
    ######################
    
    @property
    def shape(self) -> tuple:
        """
        Return the shape of the primary tensor.
        
        Returns:
        - tuple: The shape of the primary tensor
        """
        
        # Return the shape of the primary tensor
        return self.output.shape
    
    
    @property
    def has_aux(self) -> bool:
        """
        Check if there are auxiliary tensors.
        
        Returns:
        - bool: True if auxiliary tensors exist, False otherwise
        """
        
        # Return whether there are auxiliary tensors
        return len(self.aux) > 0
