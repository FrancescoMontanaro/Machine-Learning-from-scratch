import os
import re
import numpy as np
from typing import Optional, Any

from ..core import Tensor
from .modules_list import ModuleList
from .utils.context_manager import no_grad
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
        self._input_shape: Optional[tuple] = None # Input shape of the module
        
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
            
        # If the value is a ModuleList, add the modules to the dictionary of sub-modules
        elif isinstance(value, ModuleList):
            # Iterate over the modules in the ModuleList
            for i, module in enumerate(value.modules):
                # Add the module to the dictionary of sub-modules
                self.__dict__.setdefault("_modules", {})[f"{name}.{i}"] = module
                
                # Set a hierarchical name for the sub-module
                sub_module_name = f"{self.name}.{name}[{i}]" if self.name else f"{name}[{i}]"
                if module.name: sub_module_name += f".{module.name}"
                module.name = re.sub(r'([a-z])([A-Z])', r'\1_\2', sub_module_name.replace(" ", "_")).lower()

        elif isinstance(value, Tensor) and value.requires_grad and value.is_parameter:
            # Add the parameter to the dictionary of parameters
            self.__dict__.setdefault("_parameters", {})[name] = value
            
        # Set the attribute
        super().__setattr__(name, value)


    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Method to call the forward method of the module
        
        Parameters:
        - x (Tensor): Input tensor
        
        Returns:
        - Tensor: Output of the module after the forward pass
        """
        
        # Call the forward method
        return self.forward(x, *args, **kwargs)


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


    def parameters(self) -> list[Tensor]:
        """
        Method to collect the parameters of the module
        
        Returns:
        - dict: Dictionary with the parameters of the module
        """
        
        # Get the parameters of the module and its sub-modules
        params, _ = self._collect_state_tensors()
        
        # Return the parameters of the module
        return list(params.values())


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
    

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Abstract method to define the forward pass of the module
        
        Parameters:
        - x (Tensor): Input tensor
        
        Returns:
        - Tensor: Output of the module after the forward pass
        """
        
        ### Step 1: Lazy init ###
        
        # Check if the module is initialized
        if not self._output_shape:
            # Initialize the parameters of the module
            self._lazy_init(x, *args, **kwargs)
        
        ### Step 2: Forward pass, to be implemented in the child class ###
        
        # Call the forward method of the module
        out = self._forward(x, *args, **kwargs)
        
        ### Step 3: Update the output shape ###
        
        # Save the input and  output shape of the module
        self._output_shape = out.shape()
        self._input_shape = x.shape()
        
        # Return the output tensor
        return out
    
    
    def output_shape(self) -> Optional[tuple]:
        """
        Abstract method to return the output shape of the module
        
        Returns:
        - tuple: The shape of the output of the module if the module is initialized, None otherwise
        """
        
        # Return the output shape of the module
        return self._output_shape
            
    
    def summary(self, recursive: bool = False, is_root: bool = True, prefix: str = "") -> None:
        """
        Method to display the summary of the model.
        
        Parameters:
        - recursive (bool): 
            If False (default), prints the original table-based summary.
            If True, prints a tree by descending into submodules.
        - is_root (bool): 
            If True, prints headers/footers or the top-level node (depending on the mode).
            Internally used for recursion.
        - prefix (str): 
            Used internally to manage indentation for the tree layout.
        """

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
                
                # Extract the output shape and format it
                output_shape = module.output_shape()
                output_shape = f"({', '.join(str(dim) for dim in output_shape)})" if isinstance(output_shape, tuple) else "?"
                output_shape = format_summary_output(output_shape, 20)

                # Extract the number of parameters and format it
                num_params = module.count_params()
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
            print(f"Total trainable parameters: {self.count_params()}")
            print(f"{'-' * len(header)}")

        # If recursive, print the summary in tree format
        else:
            # For the root module, print its name/class (and optional shape/params)
            if is_root:
                # Extract the shape of the module and format it
                shape = self.output_shape()
                shape_str = f"({', '.join(str(dim) for dim in shape)})" if isinstance(shape, tuple) else "?"
                
                # Extract the number of parameters
                num_params = self.count_params()
                
                # Print the top-level module
                print(f"{self.name} ({self.__class__.__name__}) " f"[output_shape={shape_str}, params={str(num_params)}]")

            # Go through each child module and print in a tree-like structure
            modules = list(self._modules.values())
            for i, module in enumerate(modules):
                # Determine if this is the last child for tree-drawing symbols
                is_last = (i == len(modules) - 1)
                branch_symbol = "└──" if is_last else "├──"

                # Try to get shape
                shape = module.output_shape()
                shape_str = f"({', '.join(str(dim) for dim in shape)})" if isinstance(shape, tuple) else "?"

                # Try to get number of parameters
                num_params = module.count_params()

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
        """
        
        # Extract the parameters and buffers of the module
        params, buffers = self._collect_state_tensors()
        
        # Create a dictionary to store the state of the module
        state_dict = {
            **{"__input_shape__": self._input_shape},
            **{name: param.data for name, param in params.items()},
            **{name: buffer.data for name, buffer in buffers.items()}
        }
        
        # Create the output directory if it doesn't exist
        if os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the state of the module to a file
        np.savez(path, **state_dict)

        
    def load(self, path: str) -> None:
        """
        Method to load the state of the module from a file
        
        Parameters:
        - path (str): Path to the file containing the state of the module
        
        Raises:
        - FileNotFoundError: If the file does not exist
        - ValueError: If the input shape is not present in the state dictionary and the module is not initialized
        - ValueError: If the dimensions of the parameters or buffers do not match
        - KeyError: If the key is not found in the parameters or buffers
        """
        
        # Check if the file exists
        if not os.path.exists(path):
            # If the file does not exist, raise an error
            raise FileNotFoundError(f'The file "{path}" does not exist.')
        
        # Load the state of the module from a file
        state_dict = np.load(path)
        
        # Check if the input shape is present in the state dictionary
        if "__input_shape__" in state_dict.keys():
            # Disable gradient computation
            with no_grad():
                # Create a dummy tensor with the input shape
                dummy_tensor = Tensor(np.zeros(state_dict["__input_shape__"]))
                
                # Set the model in evaluation mode
                self.eval()
                
                # Call the forward method with the dummy tensor to initialize the parameters
                self.forward(dummy_tensor)
                
        else:
            # Check if the input shape is None
            if self._input_shape is None:
                # If the input shape is not present, raise an error
                raise ValueError(f'The input shape "__input_shape__" is not present in the state dictionary. Call the forward method before loading the state, even with a dummy tensor.')
        
        # Estrai i mapping attuali di parametri e buffer
        params, buffers = self._collect_state_tensors()
        
        # Itera sulle chiavi salvate e aggiorna i dati dei tensori corrispondenti
        for key in state_dict.files:
            # Exclude the input shape key
            if key == "__input_shape__":
                continue
            
            # Load the parameters
            if key in params.keys():
                # Chekc the dimensions of the parameter
                if params[key].data is not None and state_dict[key].shape != params[key].data.shape:
                    # Raise an error if the dimensions do not match
                    raise ValueError(f"Dimension mismatch for parameter '{key}': expected {params[key].data.shape}, got {state_dict[key].shape}")
                
                # Update the parameter data
                params[key].data = state_dict[key]
                
            # Load the buffers
            elif key in buffers.keys():
                # Check the dimensions of the buffer
                if buffers[key].data is not None and state_dict[key].shape != buffers[key].data.shape:
                    # Raise an error if the dimensions do not match
                    raise ValueError(f"Dimension mismatch for buffer '{key}': expected {buffers[key].data.shape}, got {state_dict[key].shape}")
                
                # Update the buffer data
                buffers[key].data = state_dict[key]
                
            # If the key is not found in the parameters or buffers, raise an error
            else:
                # If the key is not found in the parameters or buffers, raise an error
                raise ValueError(f"Key '{key}' not found in the module's parameters or buffers.")
                  
    
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
        
        if len(self._parameters) > 0:
            # Iterate over the parameters of the current module
            for name, param in self._parameters.items():
                # Compose the parameter name
                param_name = f"{prefix}.{name}" if prefix else name
                
                # Add the parameter to the dictionary
                params.update({param_name: param})
                
                
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
            

    def _lazy_init(self, x: Tensor, *args, **kwargs) -> None:
        """
        Abstract Method to lazily initialize the parameters of the module
        
        Parameters:
        - x (Tensor): Input tensor
        """
        
        # Check if the module is initialized
        pass
    
    
    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Abstract method to define the forward pass of the module
        This method should be implemented in the child classes
        
        Parameters:
        - x (Tensor): Input tensor
        """
        
        raise NotImplementedError("The forward method must be implemented in the child class.")