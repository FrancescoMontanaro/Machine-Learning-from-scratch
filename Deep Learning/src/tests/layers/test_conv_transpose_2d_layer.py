import os
import sys
import torch
import unittest
import numpy as np
from torch.nn import ConvTranspose2d

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.core import Tensor
from src.layers import ConvTranspose2D
from src.tests.base import Test


class TestConvTranspose2DLayer(Test):

    def setUp(self) -> None:
        """
        Set up the test case.
        This method will be called before each test.
        """

        # Create random input data (batch, height, width, channels)
        self.x_np = np.random.randn(2, 4, 4, 3).astype(np.float32)
        
        # Create the input tensors
        self.x_tensor = Tensor(self.x_np, requires_grad=True)
        self.x_torch = torch.tensor(self.x_np, requires_grad=True)

        # Layer parameters
        self.num_filters = 6
        self.kernel_size = (3, 3)
        self.stride = (2, 2)
        self.padding = (1, 1)
        self.output_padding = (1, 1)

        # Create the ConvTranspose2D layers
        self.conv_custom = ConvTranspose2D(
            num_filters = self.num_filters, 
            kernel_size = self.kernel_size, 
            stride = self.stride,
            padding = self.padding,
            output_padding = self.output_padding
        )
        self.conv_torch = ConvTranspose2d(
            in_channels = self.x_np.shape[3], 
            out_channels = self.num_filters, 
            kernel_size = self.kernel_size,
            stride = self.stride,
            padding = self.padding,
            output_padding = self.output_padding
        )

        # Initialize the custom layer
        self.conv_custom.eval()
        self.conv_custom(self.x_tensor)

        # Copy the weights and bias from the custom layer to the PyTorch layer
        with torch.no_grad():
            # Extract the weights and bias from the custom layer
            # Custom: (in_channels, out_channels, kH, kW)
            # PyTorch: (in_channels, out_channels, kH, kW) - same format!
            w_custom = torch.from_numpy(self.conv_custom.filters.data).float()
            b_custom = torch.from_numpy(self.conv_custom.bias.data).float()
            
            # Copy the weights and bias to the PyTorch layer
            self.conv_torch.weight.copy_(w_custom)
            if self.conv_torch.bias is not None: 
                self.conv_torch.bias.copy_(b_custom) 


    def test_conv_transpose_2d_forward(self) -> None:
        """
        Test to verify that the forward pass of the ConvTranspose2D layer is consistent with PyTorch.
        """
        
        # Forward pass
        # Custom: (batch, H, W, C) -> (batch, H', W', C')
        y_custom = self.conv_custom(self.x_tensor)
        
        # PyTorch: (batch, C, H, W) -> (batch, C', H', W')
        y_torch = self.conv_torch(self.x_torch.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Compare the forward pass results
        self.assertTrue(
            np.allclose(y_custom.data, y_torch.detach().numpy(), atol=1e-5),
            msg=(
                f"❌ Forward outputs differ beyond tolerance!\n"
                f"Custom shape: {y_custom.data.shape}\n"
                f"Torch shape: {y_torch.detach().numpy().shape}\n"
                f"Max diff: {np.max(np.abs(y_custom.data - y_torch.detach().numpy()))}"
            )
        )


    def test_conv_transpose_2d_backward(self) -> None:
        """
        Test to verify that the backward pass of the ConvTranspose2D layer is consistent with PyTorch.
        """
        
        # Forward pass
        y_custom = self.conv_custom(self.x_tensor)
        y_torch = self.conv_torch(self.x_torch.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # Define a simple loss (sum of all elements) and perform the backward pass
        loss_custom = y_custom.sum()
        loss_torch = y_torch.sum()
        loss_custom.backward()
        loss_torch.backward()

        # Check if the gradients are not None
        self.assertIsNotNone(self.x_tensor.grad, "Custom Tensor grad is None")
        self.assertIsNotNone(self.x_torch.grad, "PyTorch Tensor grad is None")
        
        # Check if the gradients are not None
        if self.x_tensor.grad is None or self.x_torch.grad is None:
            self.fail("Gradients are None!")

        # Compare the backward gradients
        self.assertTrue(
            np.allclose(self.x_tensor.grad, self.x_torch.grad.numpy(), atol=1e-5),
            msg=(
                f"❌ Backward gradients differ beyond tolerance!\n"
                f"Max diff: {np.max(np.abs(self.x_tensor.grad - self.x_torch.grad.numpy()))}"
            )
        )


    def test_conv_transpose_2d_output_shape(self) -> None:
        """
        Test that the output shape is correct based on the formula:
        out = (in - 1) * stride - 2 * padding + kernel_size + output_padding
        """
        
        # Calculate expected output shape
        in_height, in_width = self.x_np.shape[1], self.x_np.shape[2]
        expected_height = (in_height - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0] + self.output_padding[0]
        expected_width = (in_width - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1] + self.output_padding[1]
        
        # Forward pass
        y_custom = self.conv_custom(self.x_tensor)
        
        # Check shape
        self.assertEqual(y_custom.shape[1], expected_height, f"Height mismatch: {y_custom.shape[1]} vs {expected_height}")
        self.assertEqual(y_custom.shape[2], expected_width, f"Width mismatch: {y_custom.shape[2]} vs {expected_width}")
        self.assertEqual(y_custom.shape[3], self.num_filters, f"Channels mismatch: {y_custom.shape[3]} vs {self.num_filters}")


    def test_conv_transpose_2d_simple_case(self) -> None:
        """
        Test with simple parameters (stride=1, no padding).
        """
        
        # Simple input
        x_np = np.random.randn(1, 4, 4, 2).astype(np.float32)
        x_tensor = Tensor(x_np, requires_grad=True)
        x_torch = torch.tensor(x_np, requires_grad=True)
        
        # Create layers with simple parameters
        conv_custom = ConvTranspose2D(num_filters=3, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        conv_torch = ConvTranspose2d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=0)
        
        # Initialize custom layer
        conv_custom.eval()
        conv_custom(x_tensor)
        
        # Copy weights
        with torch.no_grad():
            w_custom = torch.from_numpy(conv_custom.filters.data).float()
            b_custom = torch.from_numpy(conv_custom.bias.data).float()
            conv_torch.weight.copy_(w_custom)
            if conv_torch.bias is not None:
                conv_torch.bias.copy_(b_custom)
        
        # Forward pass
        y_custom = conv_custom(x_tensor)
        y_torch = conv_torch(x_torch.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # Compare
        self.assertTrue(
            np.allclose(y_custom.data, y_torch.detach().numpy(), atol=1e-5),
            msg=f"Simple case forward mismatch. Max diff: {np.max(np.abs(y_custom.data - y_torch.detach().numpy()))}"
        )


if __name__ == '__main__':
    unittest.main()
