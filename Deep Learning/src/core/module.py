import re
from typing import Optional, Any

from ..core import Tensor
from .modules_list import ModuleList
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
        
        # Save the output shape of the module
        self._output_shape = out.shape()
        
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
                # Composing the module name
                module_name = f"{module.name} ({module.__class__.__name__})"

                # Composing the output shape
                output_shape = "?"
                try:
                    # Get the output shape of the module
                    output_shape = module.output_shape()
                    
                    # Format the output shape
                    output_shape = (f"({', '.join(str(dim) for dim in output_shape)})" if output_shape else "?")
                except:
                    pass

                # Composing the number of parameters
                num_params = "?"
                try:
                    # Get the number of parameters of the module
                    num_params = module.count_params()
                except:
                    pass

                # format the output
                module_name = format_summary_output(module_name, 50) + " " * 5
                output_shape = format_summary_output(str(output_shape), 20)
                num_params = format_summary_output(str(num_params), 20)

                # Display the module information
                print(f"{module_name:<55}{str(output_shape):<20}{str(num_params):<20}")
                
                # Print separator if not the last module
                if idx < len(modules) - 1:
                    header = f"{'Module (type)':<55}{'Output Shape':<20}{'Trainable params #':<20}"
                    print(f"{'-' * len(header)}")

            # Compute the total number of parameters
            total_params = "?"
            try:
                total_params = self.count_params()
            except:
                pass

            # Display the footer 
            header = f"{'Module (type)':<55}{'Output Shape':<20}{'Trainable params #':<20}"
            print(f"{'=' * len(header)}")
            print(f"Total trainable parameters: {total_params}")
            print(f"{'-' * len(header)}")

        # If recursive, print the summary in tree format
        else:
            # For the root module, print its name/class (and optional shape/params)
            if is_root:
                # Try to get shape
                shape_str = "?"
                try:
                    # Get the output shape of the module
                    shape = self.output_shape()
                    
                    # Format the output shape
                    shape_str = f"({', '.join(str(dim) for dim in shape)})" if shape else "?"
                except:
                    pass
                
                # Try to get number of parameters
                num_params_str = "?"
                try:
                    num_params_str = str(self.count_params())
                except:
                    pass
                
                # Print the top-level module
                print(f"{self.name} ({self.__class__.__name__}) " f"[output_shape={shape_str}, params={num_params_str}]")

            # Go through each child module and print in a tree-like structure
            modules = list(self._modules.values())
            for i, module in enumerate(modules):
                # Determine if this is the last child for tree-drawing symbols
                is_last = (i == len(modules) - 1)
                branch_symbol = "└──" if is_last else "├──"

                # Try to get shape
                shape_str = "?"
                try:
                    # Get the output shape of the module
                    shape = module.output_shape()
                    
                    # Format the output shape
                    shape_str = f"({', '.join(str(dim) for dim in shape)})" if shape else "?"
                except:                
                    pass

                # Try to get number of parameters
                num_params_str = "?"
                try:
                    num_params_str = str(module.count_params())
                except:
                    pass

                # Print this module line
                print(f"{prefix}{branch_symbol} {module.name} ({module.__class__.__name__}) " f"[output_shape={shape_str}, params={num_params_str}]")

                # If the module has children, recurse
                if len(module._modules) > 0:
                    # Update the prefix for the child
                    child_prefix = prefix + ("    " if is_last else "│   ")
                    module.summary(recursive=True, is_root=False, prefix=child_prefix)
                    
    
    #########################
    ### Protected methods ###
    #########################
            
            
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