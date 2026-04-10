from typing import List
from dataclasses import dataclass, field

from ..config import ModelConfig


@dataclass
class ResNetConfig(ModelConfig):
    """
    Configuration for the ResNet architecture.

    Attributes:
    - layers (List[int]): Number of residual blocks per group. Default: [3, 4, 6, 3] (ResNet-34).
    - num_classes (int): Number of output classes. Default: 10.
    """

    layers: List[int] = field(default_factory=lambda: [3, 4, 6, 3])
    num_classes: int = 10

    def __post_init__(self) -> None:
        """
        Post-initialization checks.

        Raises:
        - AssertionError: If the number of layer groups is not exactly 4.
        - AssertionError: If any layer group has zero blocks.
        """

        assert len(self.layers) == 4, "ResNet requires exactly 4 layer groups."
        assert all(n > 0 for n in self.layers), "Each layer group must have at least 1 block."
