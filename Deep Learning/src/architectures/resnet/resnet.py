from typing import List

from ...activations import ReLU
from .config import ResNetConfig
from ..sequential import Sequential
from ...core import Tensor, Module, ModuleList
from ...layers import Conv2D, BatchNormalization, MaxPool2D, Dense, ResidualBlock


class ConvBlock(Module):

    ### Magic methods ###

    def __init__(self, out_channels: int, stride: int = 1, *args, **kwargs) -> None:
        """
        Basic convolutional block used in ResNet.
        
        Parameters:
        - out_channels (int): Number of output channels for the convolutional layers.
        - stride (int): Stride for the first convolutional layer. Default is 1 (no downsampling).
        """

        # Initialize the base Module class
        super().__init__(*args, **kwargs)

        # Define the two convolutional layers with batch normalization.
        self.conv1 = Conv2D(out_channels, (3, 3), padding="same", stride=(stride, stride))
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(out_channels, (3, 3), padding="same", stride=(1, 1))
        self.bn2 = BatchNormalization()
    

    ### Protected methods ###

    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass of the convolutional block.
        
        Parameters:
        - x (Tensor): Input tensor of shape (B, H, W, C).
        
        Returns:
        - Tensor: Output tensor of shape (B, H, W, out_channels).
        """

        # First convolutional layer + BN + ReLU
        x = self.conv1(x).output
        x = self.bn1(x).output
        x = x.relu()

        # Second convolutional layer + BN (no activation here)
        x = self.conv2(x).output
        x = self.bn2(x).output

        # Return the output of the block before adding the shortcut and applying the activation function.
        return x


class ProjectionResidualBlock(Module):
    
    ### Magic methods ###

    def __init__(self, out_channels: int, stride: int = 1, *args, **kwargs) -> None:
        """
        Residual block with a projection shortcut, used when the input and output dimensions differ.
        
        Parameters:
        - out_channels (int): Number of output channels for the convolutional layers.
        - stride (int): Stride for the first convolutional layer. Default is 1 (no downsampling).
        """

        # Initialize the base Module class
        super().__init__(*args, **kwargs)

        # Define the convolutional block for the main path and the projection shortcut for the skip connection.
        self.conv_block = ConvBlock(out_channels, stride)
        self.shortcut_conv = Conv2D(out_channels, (1, 1), padding="valid", stride=(stride, stride))
        self.shortcut_bn = BatchNormalization()


    ### Protected methods ###

    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass of the projection residual block.
        
        Parameters:
        - x (Tensor): Input tensor of shape (B, H, W, C).
        
        Returns:
        - Tensor: Output tensor of shape (B, H, W, out_channels).
        """

        # Compute the main path through the convolutional block
        out = self.conv_block(x).output
        shortcut = self.shortcut_conv(x).output
        shortcut = self.shortcut_bn(shortcut).output
        out = out + shortcut
        out = out.relu()

        # Return the output of the block after adding the shortcut and applying the activation function.
        return out


class ResNetModule(Module):

    ### Magic methods ###

    def __init__(self, config: ResNetConfig = ResNetConfig(), *args, **kwargs) -> None:
        """
        ResNet architecture module containing all layers and blocks.
        
        Parameters:
        - config (ResNetConfig): Architecture configuration.
        """

        # Initialize the base Module class
        super().__init__(*args, **kwargs)

        # Initial feature extraction: 7x7 conv + BN + MaxPool
        self.conv1 = Conv2D(64, (7, 7), padding="same", stride=(2, 2))
        self.bn1 = BatchNormalization()
        self.maxpool = MaxPool2D(size=(3, 3), stride=(2, 2), padding="same")

        # Residual layer groups
        self.layer0 = ModuleList(self._make_layer(64,  config.layers[0], stride=1, in_channels=64))
        self.layer1 = ModuleList(self._make_layer(128, config.layers[1], stride=2, in_channels=64))
        self.layer2 = ModuleList(self._make_layer(256, config.layers[2], stride=2, in_channels=128))
        self.layer3 = ModuleList(self._make_layer(512, config.layers[3], stride=2, in_channels=256))

        # Classification head
        self.fc = Dense(config.num_classes)


    ### Protected methods ###

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int, in_channels: int) -> List[Module]:
        """
        Build a group of residual blocks.

        The first block uses a projection shortcut when spatial downsampling or channel
        change is required; subsequent blocks use the identity shortcut.

        Parameters:
        - out_channels (int): Number of output channels for every block in this group.
        - num_blocks (int): Number of residual blocks in this group.
        - stride (int): Stride for the first block (1 = no downsampling, 2 = halve spatial dims).
        - in_channels (int): Number of input channels coming into this group.

        Returns:
        - List[Module]: Ordered list of residual blocks for this group.
        """

        # Initialize an empty list to hold the blocks for this layer group
        blocks: List[Module] = []

        # First block: projection shortcut if spatial size or channel count changes
        if stride != 1 or in_channels != out_channels:
            blocks.append(ProjectionResidualBlock(out_channels, stride))
        else:
            blocks.append(ResidualBlock(
                block = ConvBlock(out_channels, stride=1),
                activation = ReLU()
            ))

        # Remaining blocks: identity shortcut (channels and spatial dims are consistent)
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(
                block = ConvBlock(out_channels, stride=1),
                activation = ReLU()
            ))

        # Return the list of blocks for this layer group
        return blocks
    

    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass of the ResNet architecture.
        
        Parameters:
        - x (Tensor): Input tensor of shape (B, H, W, C).
        
        Returns:
        - Tensor: Output tensor of shape (B, num_classes).
        """

        # Initial feature extraction
        x = self.conv1(x).output
        x = self.bn1(x).output
        x = x.relu()
        x = self.maxpool(x).output

        # Residual layer groups
        x = self.layer0(x=x).output
        x = self.layer1(x=x).output
        x = self.layer2(x=x).output
        x = self.layer3(x=x).output

        # Global average pooling: (B, H, W, C) -> (B, C)
        x = x.mean(axis=(1, 2))

        # Classification
        return self.fc(x).output


class ResNet(Sequential):

    ### Magic methods ###

    def __init__(self, config: ResNetConfig = ResNetConfig(), *args, **kwargs) -> None:
        """
        ResNet architecture.
        
        Parameters:
        - config (ResNetConfig): Architecture configuration.
        """

        # Initialize the Sequential with a single ResNetModule, which contains all the layers and blocks.
        super().__init__(
            modules = [ResNetModule(config, name="ResNet")],
            *args, **kwargs
        )
