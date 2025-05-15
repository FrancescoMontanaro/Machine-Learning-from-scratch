from collections import UserList
from typing import Optional, Iterable, TYPE_CHECKING

from . import Tensor
if TYPE_CHECKING: from .module import Module
from .utils.types_registry import get_module_class


class TensorsList(UserList):
    
    ### Magic Methods ###
    
    def __init__(self, tensors: Optional[Iterable[Tensor]] = None) -> None:
        """
        Class constructor
        
        Parameters:
        - tensors (Optional[Iterable[Tensor]]): An optional iterable of Tensor objects to initialize the list.
        
        Raises:
        - TypeError: If any of the elements in the list are not of type Tensor.
        """
        
        # Check if all elements in the list are of type Tensor
        if tensors is not None and not all(isinstance(tensor, Tensor) for tensor in tensors):
            raise TypeError("All elements in the list must be of type Tensor.")
        
        # Call the parent class constructor
        super().__init__()
        
        # Initialize the parent module and attribute name to None
        self._parent_module: Optional[Module] = None
        self._attribute_name: Optional[str] = None
        
        # Assign the tensors to the internal data list
        if tensors is not None:
            self.extend(tensors)
        
        
    def __setitem__(self, i, item) -> None:
        """
        Sets the item at the specified index in the list.
        
        Parameters:
        - i: The index at which to set the item.
        - item (Tensor): The Tensor to set at the specified index.
        """
        
        # Check if the item is of type Tensor
        if not isinstance(item, Tensor):
            # Raise a TypeError if the item is not a Tensor
            raise TypeError("Invalid item type. Expected Tensor.")

        # Call the parent method to set the item
        super().__setitem__(i, item)
        
        # If the parent module and attribute name are set, register the new item
        if self._parent_module and self._attribute_name:
            # Register the new item with the parent module
            # It will overwrite the old one if it exists
            self._parent_module._register_indexed_tensor(self._attribute_name, i, item)


    def __delitem__(self, i) -> None:
        """
        Deletes the item at the specified index from the list.
        
        Parameters:
        - i: The index of the item to delete.
        """
        
        # Call the parent method to delete the item
        super().__delitem__(i)
        
        # If the parent module is set, rebuild the registered tensors
        if self._parent_module:
            # Unregister the item from the parent module
            self._rebuild_registered_tensors()
        
    
    ### Public Methods ###

    def append(self, item: Tensor) -> None:
        """
        Appends a Tensor to the list.
        
        Parameters:
        - item (Tensor): The Tensor to append to the list.
        
        Raises:
        - TypeError: If the item is not of type Tensor.
        """
        
        # Check if the item is of type Tensor
        if not isinstance(item, Tensor):
            # Raise a TypeError if the item is not a Tensor
            raise TypeError("Invalid item type. Expected Tensor.")
        
        # Call the parent method to append the item
        super().append(item)
        
        # If the parent module and attribute name are set, register the new item
        if self._parent_module and self._attribute_name:
            # Compose the new index based on the current length of the data
            new_index = len(self.data) - 1
            
            # Register the new item with the parent module
            self._parent_module._register_indexed_tensor(self._attribute_name, new_index, item)
            

    def insert(self, i: int, item: Tensor) -> None:
        """
        Inserts a Tensor at the specified index in the list.
        
        Parameters:
        - i (int): The index at which to insert the Tensor.
        - item (Tensor): The Tensor to insert into the list.
        
        Raises:
        - TypeError: If the item is not of type Tensor.
        """
        
        # Check if the item is of type Tensor
        if not isinstance(item, Tensor):
            # Raise a TypeError if the item is not a Tensor
            raise TypeError("Invalid item type. Expected Tensor.")
        
        # Call the parent method to insert the item
        super().insert(i, item)
        
        # If the parent module is set, register the new item by rebuilding the list
        if self._parent_module:
            # Register the new item with the parent module
            self._rebuild_registered_tensors()
            

    def pop(self, i: int = -1) -> Tensor:
        """
        Pops the item at the specified index from the list.
        
        Parameters:
        - i (int): The index of the item to pop. Defaults to -1 (last item).
        
        Returns:
        - Tensor: The popped Tensor.
        """
        
        # Call the parent method to pop the item
        item = super().pop(i)
        
        # If the parent module is set, rebuild the registered tensors
        if self._parent_module:
            # Unregister the item from the parent module
            self._rebuild_registered_tensors()
            
        # Return the popped item
        return item


    def extend(self, other: Iterable[Tensor]) -> None:
        """
        Extends the list with the items from another iterable.
        
        Parameters:
        - other (Iterable[Tensor]): The iterable of Tensors to extend the list with.
        
        Raises:
        - TypeError: If any of the items in the iterable are not of type Tensor.
        """
        
        # Create a list to store the validated items
        validated_items = []
        
        # Iterate over the items in the other iterable
        for item in other:
            # Check if the item is of type Tensor
            if not isinstance(item, Tensor):
                # Raise a TypeError if the item is not a Tensor
                raise TypeError("Invalid item type. Expected Tensor.")
            
            # Append the validated item to the list
            validated_items.append(item)
        
        # Call the parent method to extend the list
        super().extend(validated_items)
        
        # If the parent module and attribute name are set, register the new items
        if self._parent_module:
            # Iterate over the validated items and register them with the parent module
            self._rebuild_registered_tensors()
            

    def clear(self) -> None:
        """
        Clears the list and unregisters all items from the parent module.
        """
        
        # Call the parent method to clear the list
        super().clear()
        
        # If the parent module is set, rebuild the registered tensors
        if self._parent_module and self._attribute_name:
            # Unregister all items from the parent module
            self._parent_module._clear_indexed_tensors(self._attribute_name)
    
    
    ### Protected Methods ###
    
    def _assign_to_module(self, parent_module: 'Module', attribute_name: str) -> None:
        """
        Assigns the TensorsList to a parent module and attribute name.
        
        Parameters:
        - parent_module (Module): The parent module to which the TensorsList will be assigned.
        - attribute_name (str): The name of the attribute in the parent module.
        
        Raises:
        - TypeError: If parent_module is not an instance of Module or if attribute_name is not a string.
        """
        
        # Lazy get the class of the parent module
        ModuleCls = get_module_class()
        
        # Check if the parent module is an instance of Module
        if not isinstance(parent_module, ModuleCls):
            raise TypeError("parent_module must be an instance of Module.")
        
        # Check if the attribute name is a string
        if not isinstance(attribute_name, str):
            raise TypeError("attribute_name must be a string.")
        
        # Assign the parent module and attribute name
        self._parent_module = parent_module
        self._attribute_name = attribute_name
        
        # Register the TensorsList with the parent module
        self._rebuild_registered_tensors()
        
    
    def _rebuild_registered_tensors(self) -> None:
        """
        Rebuilds the registered tensors in the parent module.
        """
        
        # Check if the parent module and attribute name are set
        if not self._parent_module or not self._attribute_name:
            # No parent module or attribute name is set, so nothing to do
            return

        # Clear the existing indexed tensors in the parent module
        self._parent_module._clear_indexed_tensors(self._attribute_name)
        
        # Iterate over the tensors in the list and register them with the parent module
        for i, t in enumerate(self.data):
            # Check if the parameter is an instance of Tensor
            if not isinstance(t, Tensor):
                # Raise a TypeError if the parameter is not a Tensor
                raise TypeError(f"Element at index {i} is not a Tensor: {t}")
            
            # Register the tensors with the parent module
            self._parent_module._register_indexed_tensor(self._attribute_name, i, t)