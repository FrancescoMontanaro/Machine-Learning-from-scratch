import os
import re
import numpy as np
import dill as pickle
from typing import Optional, Any, Tuple, Union, TYPE_CHECKING

from .tensor import Tensor
from .tensors_list import TensorsList
from .utils.data_analysis import format_summary_output


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
        self.name: str = name or self.__class__.__name__ # Name of the module
        self.training: bool = True # Flag to check if the module is in training mode
        self._output_shape: Optional[tuple] = None # Output shape of the module
        
        # Initialize the dictionaries for the parameters and sub-modules
        self._parameters: dict[str, Tensor] = {} # Dictionary for the parameters of the module
        self._modules: dict[str, 'Module'] = {} # Dictionary for the sub-modules of the module
        self._buffers: dict[str, Tensor] = {} # Dictionary for the buffers of the module


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
            value.name = re.sub(r'([a-z])([A-Z])', r'\1_\2', value.name.replace(" ", "_")).lower()

        elif isinstance(value, Tensor) and value.requires_grad and value.is_parameter:
            # Add the parameter to the dictionary of parameters
            self.__dict__.setdefault("_parameters", {})[name] = value
            
        elif isinstance(value, TensorsList):
            # Assign the list to the module
            value._assign_to_module(parent_module=self, attribute_name=name)
            
        elif isinstance(value, (list)):
            # Print a warning to suggest using TensorsList
            print(f"Warning: {name} is a list of tensors. Consider using TensorsList instead for better and dynamic management.")
            
            # Iterate over the list of tensors
            for i, v in enumerate(value):
                # Check if the tensor is a parameter
                if isinstance(v, Tensor) and v.requires_grad and v.is_parameter:
                    # Add the parameter to the dictionary of parameters
                    self.__dict__.setdefault("_parameters", {})[f"{name}_{i}"] = v
            
        # Set the attribute
        super().__setattr__(name, value)


    def __call__(self, *args, **kwargs) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Method to call the forward method of the module
        
        Parameters:
        - x (Tensor): Input tensor
        
        Returns:
        - Union[Tensor, Tuple[Tensor, ...]]: Output of the module after the forward pass
        """
        
        # Check if more than one positional argument is provided
        if len(args) > 1:
            raise ValueError("Only one positional argument is allowed, and it will be treated as the 'x' input tensor.")
        
        # If only one positional argument is provided, treat it as 'x'
        if len(args) == 1:
            # Validate that the positional argument is a Tensor and 'x' is not already in kwargs
            if not isinstance(args[0], Tensor):
                raise ValueError("The single positional argument must be a Tensor.")
            
            # If 'x' is already in kwargs, raise an error
            if 'x' in kwargs:
                raise ValueError("Cannot provide 'x' both as positional and keyword argument.")
            
            # Convert the positional argument to a keyword argument
            kwargs['x'] = args[0]
        
        # Call the forward method
        return self.forward(**kwargs)


    ######################
    ##### Properties #####
    ######################
    
    @property
    def parameters(self) -> list[Tensor]:
        """
        Property to get the parameters of the module
        
        Returns:
        - dict: Dictionary with the parameters of the module
        """
        
        # Get the parameters of the module and its sub-modules
        params, _ = self._collect_state_tensors()
        
        # Return the parameters of the module
        return list(params.values())
    
    
    @property
    def params_count(self) -> int:
        """
        Property to get the number of parameters in the module
        
        Returns:
        - int: Number of parameters in the module
        """
        
        # Initialize the counter for the number of parameters
        total = 0
        
        # Iterate over the parameters of the module
        for param in self.parameters:
            # Add the number of parameters in the current parameter
            total += param.data.size if param.is_parameter else 0
            
        # Return the total number of parameters
        return total


    @property
    def is_initialized(self) -> bool:
        """
        Property to check if the module is initialized
        
        Returns:
        - bool: True if the module is initialized, False otherwise
        """
        
        # Check if the output shape of the module is set
        return self._output_shape is not None
    
    
    @property
    def output_shape(self) -> Optional[tuple]:
        """
        Abstract method to return the output shape of the module
        
        Returns:
        - tuple: The shape of the output of the module if the module is initialized, None otherwise
        """
        
        # Return the output shape of the module
        return self._output_shape
    

    ######################
    ### Public methods ###
    ######################
    
    def register_buffer(self, name: str, tensor: Tensor) -> None:
        """
        Method to register a buffer in the module
        
        Parameters:
        - name (str): Name of the buffer
        - tensor (Tensor): Tensor to register as buffer
        """
        
        # Set the requires_grad attribute of the tensor to False
        tensor.requires_grad = False
        tensor.is_parameter = False
        
        # Add the tensor to the dictionary of buffers
        self._buffers[name] = tensor
        
        # Set the attribute of the module
        setattr(self, name, tensor)
        

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


    def reset_cache(self) -> None:
        """
        Method to reset the cache of the module
        """
        
        # Iterate over the sub-modules of the module
        for module in self._modules.values():
            # Check if the sub-module has a reset_cache method
            if hasattr(module, "reset_cache") and callable(getattr(module, "reset_cache")):
                # Call the reset_cache method of the sub-module
                module.reset_cache()


    def forward(self, *args, **kwargs) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Abstract method to define the forward pass of the module
        
        Returns:
        - Union[Tensor, Tuple[Tensor, ...]]: Output of the module after the forward pass
        """
        
        ### Step 1: Lazy init ###
        
        # Check if the module is initialized
        if not self._output_shape:
            # Initialize the parameters of the module
            self._lazy_init(*args, **kwargs)
        
        ### Step 2: Forward pass, to be implemented in the child class ###
        
        # Call the forward method of the module
        out = self._forward(*args, **kwargs)
        
        ### Step 3: Update the output shape ###
        
        # Save the input and  output shape of the module
        self._output_shape = tuple(o.shape for o in out) if isinstance(out, tuple) else out.shape
        
        # Return the output tensor
        return out
            
    
    def summary(self, recursive: bool = False, is_root: bool = True, prefix: str = "") -> None:
        """
        Method to display the summary of the model.
        
        Parameters:
        - recursive (bool): If False (default), prints the original table-based summary. If True, prints a tree by descending into submodules.
        - is_root (bool): If True, prints headers/footers or the top-level node (depending on the mode). Internally used for recursion.
        - prefix (str): Used internally to manage indentation for the tree layout.
        """
        
        def _format_shape(shape: Optional[tuple]) -> str:
            """
            Helper to format output shape, handling both single and multiple outputs.
            
            Parameters:
            - shape: The output shape(s) to format, which can be a tuple for single output or a tuple of tuples for multiple outputs.
            
            Returns:
            - str: A formatted string representing the output shape(s).
            """

            # If shape is None, return "?" to indicate unknown shape
            if shape is None:
                return "?"
            
            # Check if this is a tuple of shapes (multiple outputs)
            # A tuple of shapes will have tuple elements, while a single shape has int elements
            if isinstance(shape, tuple) and len(shape) > 0 and isinstance(shape[0], tuple):
                # Multiple output shapes: format each and join with ", "
                return "[" + ", ".join(f"({', '.join(str(d) for d in s)})" for s in shape) + "]"
            else:
                # Single output shape
                return f"({', '.join(str(dim) for dim in shape)})" if isinstance(shape, tuple) else "?"

        # If NOT recursive, print the summary in tabular format
        if not recursive:
            # Diaplay the header
            print(f"\n{self.name}\n")
            header = f"{'Module (type)':<55}{'Output Shape':<20}{'Trainable params #':<20}"
            print(f"{'-' * len(header)}")
            print(header)
            print(f"{'=' * len(header)}")

            # Iterate over the modules
            modules = list(self._modules.values())
            for idx, module in enumerate(modules):
                # Composing the module name and format it
                module_name = f"{module.name} ({module.__class__.__name__})"
                module_name = format_summary_output(module_name, 50) + " " * 5
                
                # Extract the output shape and format it using helper function
                output_shape = _format_shape(module.output_shape)
                output_shape = format_summary_output(output_shape, 20)

                # Extract the number of parameters and format it
                num_params = module.params_count
                num_params = format_summary_output(str(num_params), 20)

                # Display the module information
                print(f"{module_name:<55}{str(output_shape):<20}{str(num_params):<20}")
                
                # Print separator if not the last module
                if idx < len(modules) - 1:
                    header = f"{'Module (type)':<55}{'Output Shape':<20}{'Trainable params #':<20}"
                    print(f"{'-' * len(header)}")

            # Display the footer 
            header = f"{'Module (type)':<55}{'Output Shape':<20}{'Trainable params #':<20}"
            print(f"{'=' * len(header)}")
            print(f"Total trainable parameters: {self.params_count}")
            print(f"{'-' * len(header)}")

        # If recursive, print the summary in tree format
        else:
            # For the root module, print its name/class (and optional shape/params)
            if is_root:
                # Extract the shape of the module and format it using helper function
                shape_str = _format_shape(self.output_shape)
                
                # Extract the number of parameters
                num_params = self.params_count
                
                # Print the top-level module
                print(f"{self.name} ({self.__class__.__name__}) " f"[output_shape={shape_str}, params={str(num_params)}]")

            # Go through each child module and print in a tree-like structure
            modules = list(self._modules.values())
            for i, module in enumerate(modules):
                # Determine if this is the last child for tree-drawing symbols
                is_last = (i == len(modules) - 1)
                branch_symbol = "└──" if is_last else "├──"

                # Format shape using helper function
                shape_str = _format_shape(module.output_shape)

                # Try to get number of parameters
                num_params = module.params_count

                # Print this module line
                print(f"{prefix}{branch_symbol} {module.name} ({module.__class__.__name__}) " f"[output_shape={shape_str}, params={str(num_params)}]")

                # If the module has children, recurse
                if len(module._modules) > 0:
                    # Update the prefix for the child
                    child_prefix = prefix + ("    " if is_last else "│   ")
                    module.summary(recursive=True, is_root=False, prefix=child_prefix)
      

    def save(self, path: str) -> None:
        """
        Method to save the state of the module to a file
        
        Parameters:
        - path (str): Path to the file where the state of the module will be saved
        
        Raises:
        - ValueError: If the path is not a directory
        """
        
        # If the path exists, check if it is a directory
        if os.path.exists(path) and not os.path.isdir(path):
            # Raise an error if the path is not a directory
            raise ValueError(f"Path '{path}' must be a directory.")
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the module to a file
        self.save_module(os.path.join(path, f"module.pkl"))
        
        # Save the weights of the module to a file
        self.save_weights(os.path.join(path, f"params.npz"))
        
        
    @classmethod
    def load(cls, path: str) -> 'Module':
        """
        Method to load the state of the module from a file
        
        Parameters:
        - path (str): Path to the file containing the state of the module
        """
        
        # Check if the path exists and is a directory
        if not os.path.exists(path) or not os.path.isdir(path):
            # Raise an error if the path is not a directory
            raise ValueError(f"Path '{path}' must be a directory.")
        
        # Load the module from a file
        module = cls.load_module(os.path.join(path, f"module.pkl"))
        
        # Load the weights of the module from a file
        module.load_weights(os.path.join(path, f"params.npz"))
        
        # Return the loaded module
        return module
        
        
    def save_weights(self, path: str) -> None:
        """
        Method to save the weights and buffers of the module to a file
        
        Parameters:
        - path (str): Path to the file where the weights of the module will be saved
        """
        
        # Extract the parameters and buffers of the module
        params, buffers = self._collect_state_tensors()
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
            
        # Save the state of the module to a file
        np.savez(
            file = path,
            allow_pickle = True,
            **{f"parameters.{p_name}": p.data for p_name, p in params.items()},
            **{f"buffers.{b_name}": b.data for b_name, b in buffers.items()}
        )
        
    
    def load_weights(self, path: str) -> None:
        """
        Method to load the weights and buffers of the module from a file
        
        Parameters:
        - path (str): Path to the file containing the weights of the module
        """
        
        # Load the state of the module from a file
        data = np.load(path, allow_pickle=True)
        
        # Extract the parameters and buffers of the module
        params, buffers = self._collect_state_tensors()
        
        # Iterate over the params of the module
        for param_name, param in params.items():
            # Check if the parameter is present in the state dictionary
            if f"parameters.{param_name}" not in data.files:
                # Raise an error if the parameter is not found
                raise KeyError(f"Key '{param_name}' not found in the state dictionary.")
                
            # Update the parameter data
            param.data = data[f"parameters.{param_name}"]
            
        # Iterate over the buffers of the module
        for buffer_name, buffer in buffers.items():
            # Check if the buffer is present in the state dictionary
            if f"buffers.{buffer_name}" not in data.files:
                # Raise an error if the buffer is not found
                raise KeyError(f"Key '{buffer_name}' not found in the state dictionary.")
                
            # Update the buffer data
            buffer.data = data[f"buffers.{buffer_name}"]


    def save_module(self, path: str) -> None:
        """
        Method to save the module to a file
        
        Parameters:
        - path (str): Path to the file where the module will be saved
        """

        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the state of the module to a file
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
     
    @staticmethod       
    def load_module(path: str) -> 'Module':
        """
        Method to load the module from a file
        
        Parameters:
        - path (str): Path to the file containing the module
        
        Returns:
        - Module: Loaded module
        """
        
        # Load the module from a file
        with open(path, 'rb') as f:
            module = pickle.load(f)
            
        # Return the loaded module
        return module

    
    #########################
    ### Protected methods ###
    #########################
    
    def _collect_state_tensors(self, prefix: str = "") -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """
        Method to traverse the computational tree of the module and collect the parameters and buffers
        
        Parameters:
        - prefix (str): Prefix for the parameter names
        
        Returns:
        - tuple[dict[str, Tensor], dict[str, Tensor]]: a tuple containing two dictionaries of parameters and buffers
        """
        
        # Create dictionaries to store the parameters and buffers
        params, buffers = {}, {}
        
        # Check if the module has parameters
        if len(self._parameters) > 0:
            # Iterate over the parameters of the current module
            for name, param in self._parameters.items():
                # Compose the parameter name
                param_name = f"{prefix}.{name}" if prefix else name
                
                # Add the parameter to the dictionary
                params.update({param_name: param})
                
        # Check if the module has buffers     
        if len(self._buffers) > 0:
            # Iterate over the buffers of the current module
            for name, buffer in self._buffers.items():
                # Compose the buffer name
                buffer_name = f"{prefix}.{name}" if prefix else name
                
                # Add the buffer to the dictionary
                buffers.update({buffer_name: buffer})
                
        # Check if the module has sub-modules
        if len(self._modules) > 0:
            # Iterate over the modules of the current module
            for name, module in self._modules.items():
                # Compose the module name
                module_name = f"{prefix}.{name}" if prefix else name
                
                # Recursively call the method for sub-modules
                sub_params, sub_buffers = module._collect_state_tensors(prefix=module_name)
                
                # Update the dictionaries with the parameters and buffers of the sub-module
                params.update(sub_params)
                buffers.update(sub_buffers)
                
        # Return the parameters and buffers 
        return params, buffers


    def _lazy_init(self, *args, **kwargs) -> None:
        """
        Abstract Method to lazily initialize the parameters of the module
        
        Parameters:
        - x (Tensor): Input tensor
        """
        
        # Check if the module is initialized
        pass
    
    
    def _forward(self, *args, **kwargs) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Abstract method to define the forward pass of the module
        This method should be implemented in the child classes
        
        Parameters:
        - x (Tensor): Input tensor
        
        Returns:
        - Union[Tensor, Tuple[Tensor, ...]]: Output tensor(s) of the forward pass
        """
        
        raise NotImplementedError("The forward method must be implemented in the child class.")


    def _clear_indexed_tensors(self, list_name: str) -> None:
        """
        Remove all parameters from a list of parameters.
        
        Parameters:
        - list_name (str): Name of the list of parameters to clear.
        """
        
        # Extrae the parameters from the module
        _params = self.__dict__.get("_parameters", {})
        
        # Remove all parameters that start with the list name
        keys_to_remove = [k for k in _params if k.startswith(f"{list_name}_")]
        
        # Remove the parameters from the dictionary
        for k in keys_to_remove:
            # Remove the parameter from the dictionary
            del _params[k]


    def _register_indexed_tensor(self, list_name: str, index: int, parameter: Tensor) -> None:
        """
        Register a single tensor in a list of parameters with a specific index.
        
        Parameters:
        - list_name (str): Name of the list of parameters.
        - index (int): Index of the parameter in the list.
        - parameter (Tensor): Parameter to register.
        """
        
        # Ensure that the parameters list exists
        _params = self.__dict__.setdefault("_parameters", {})
        
        # Check if the parameter is a valid tensor
        if not (isinstance(parameter, Tensor)):
            raise TypeError(
                f"Parameter '{parameter}' is not a valid tensor. "
                f"Expected a Tensor, but got {type(parameter)}."
            )
            
        # Register the parameter in the dictionary
        _params[f"{list_name}_{index}"] = parameter


class SingleOutputModule(Module):
    """
    Base class for modules that always return a single Tensor.
    """
    
    if TYPE_CHECKING:
        def __call__(self, *args, **kwargs) -> Tensor: ...
        def _forward(self, *args, **kwargs) -> Tensor: ...
        def forward(self, *args, **kwargs) -> Tensor: ...


class MultiOutputModule(Module):
    """
    Base class for modules that always return a tuple of Tensors.
    """
    
    if TYPE_CHECKING:
        def __call__(self, *args, **kwargs) -> Tuple[Tensor, ...]: ...
        def _forward(self, *args, **kwargs) -> Tuple[Tensor, ...]: ...
        def forward(self, *args, **kwargs) -> Tuple[Tensor, ...]: ...