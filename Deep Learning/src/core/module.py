from typing import Optional, Any

from ..core import Tensor


class Module:
    
    #####################
    ### Magic methods ###
    #####################
    
    def __init__(self, name: Optional[str] = None) -> None:
        """
        Class constructor of the abstract class Module
        
        Parameters:
        - name (Optional[str]): Name of the module
        """
        
        # Set the class attributes
        self.name = name # Name of the module
        self.training = True # Flag to check if the module is in training mode
        self.initialized = False # Flag to check if the module is initialized
        
        # Initialize the dictionaries for the parameters and sub-modules
        self._parameters: dict[str, Tensor] = {} # Dictionary for the parameters of the module
        self._modules: dict[str, 'Module'] = {} # Dictionary for the sub-modules of the module


    def __setattr__(self, name: str, value: Any) -> None:
        """
        Method to set an attribute of the module
        
        Parameters:
        - name (str): Key of the attribute
        - value (Any): Value of the attribute
        """
        
        # If the value is a module, add it to the dictionary of sub-modules
        if isinstance(value, Module):
            # Add the module to the dictionary of sub-modules
            self.__dict__.setdefault("_modules", {})[name] = value
            
            # Set a hierarchical name for the sub-module
            value.name = f"{self.name}.{name}" if self.name else name

        elif isinstance(value, Tensor) and value.requires_grad and value.is_parameter:
            # Add the parameter to the dictionary of parameters
            self.__dict__.setdefault("_parameters", {})[name] = value
            
        # Set the attribute
        super().__setattr__(name, value)


    def __call__(self, x: Tensor) -> Tensor:
        """
        Method to call the forward method of the module
        
        Parameters:
        - x (Tensor): Input tensor
        
        Returns:
        - Tensor: Output of the module after the forward pass
        """
        
        # Call the forward method
        return self.forward(x)


    ######################
    ### Public methods ###
    ######################
    
    def named_parameters(self, prefix: str = "") -> dict[str, Tensor]:
        """
        Function to recursively collect the parameters of the module and its sub-modules
        
        Parameters:
        - prefix (str): Prefix for the name of the parameters
        
        Returns:
        - dict: Dictionary with the parameters of the module and its sub-modules
        """
        
        # Initialize the dictionary of parameters
        params = {}
        
        # Iterate over the parameters of the current module
        for name, param in self._parameters.items():
            # Compose the full name of the parameter
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Add the parameter to the dictionary
            params[full_name] = param
            
        # Recursively collect the parameters of the sub-modules
        for name, module in self._modules.items():
            # Compose the new prefix for the sub-module
            new_prefix = f"{prefix}.{name}" if prefix else name
            
            # Update the parameters with the parameters of the sub-module
            params.update(module.named_parameters(new_prefix))
            
        # Return the parameters
        return params

    def parameters(self) -> list[Tensor]:
        """
        Method to collect the parameters of the module
        
        Returns:
        - dict: Dictionary with the parameters of the module
        """
        
        # Return the parameters of the module
        return list(self.named_parameters().values())


    def train(self) -> None:
        """
        Method to set the module to training mode
        """
        
        # Set the module to training mode
        self.training = True
        
        # Recursively set the sub-modules to training mode
        for module in self._modules.values():
            # Set the sub-module to training mode
            module.train()


    def eval(self) -> None:
        """
        Method to set the module to evaluation mode
        """
        
        # Set the module to evaluation mode
        self.training = False
        
        # Recursively set the sub-modules to evaluation mode
        for module in self._modules.values():
            # Set the sub-module to evaluation mode
            module.eval()


    def count_params(self) -> int:
        """
        Function to count the number of parameters in the module
        
        Returns:
        - int: Number of parameters in the module
        """
        
        # Initialize the counter for the number of parameters
        total = 0
        
        # Iterate over the parameters of the module
        for param in self.parameters():
            # Add the number of parameters in the current parameter
            total += param.data.size
            
        # Return the total number of parameters
        return total
    

    def forward(self, x: Tensor) -> Tensor:
        """
        Abstract method to define the forward pass of the module
        
        Parameters:
        - x (Tensor): Input tensor
        
        Returns:
        - Tensor: Output of the module after the forward pass
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("Subclasses must implement forward()")
    
    
    def output_shape(self) -> Optional[tuple]:
        """
        Abstract method to return the output shape of the module
        
        Returns:
        - tuple: The shape of the output of the module if the module is initialized, None otherwise
        """
        
        # Check if the module is initialized  
        if not self.initialized:
            # Raise an error if the module is not initialized
            raise ValueError("Layer is not initialized. Please call the forward method with some input data.")
    
    
    def init_params(self, *args, **kwargs) -> None:
        """
        Method to initialize the parameters of the module
        In general, it is called in the first forward pass of the module to initialize the parameters
        """
        
        # Set the flag to True to indicate that the module is initialized
        self.initialized = True