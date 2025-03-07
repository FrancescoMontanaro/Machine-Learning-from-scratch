from typing import Optional

class Node:

    ### Magic methods ###

    def __init__(self, feature: int | None = None, threshold: float | None = None, left: Optional['Node'] = None, right: Optional['Node'] = None, *, value = None) -> None:
        """
        Class constructor
        
        Parameters:
        - feature (int): Feature index
        - threshold (float): Threshold value
        - left (Node): Left child node
        - right (Node): Right child node
        - value: Value of the node
        """

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    
    ### Public methods ###
    
    def is_leaf_node(self) -> bool:
        """
        Check if the node is a leaf node
        
        Returns:
        - bool: True if the node is a leaf node, False otherwise
        """
        
        return self.value is not None