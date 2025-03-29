import os
import sys
import torch
import unittest
import numpy as np
from torch.nn import Conv2d

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core import Tensor
from src.layers import Conv2D


class TestConv2DLayer(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up the test case.
        This method will be called before each test.
        """

        # Create random input data
        self.x_np = np.random.randn(1, 4, 4, 1).astype(np.float32)
        
        # Create the input tensors
        self.x_tensor = Tensor(self.x_np, requires_grad=True)
        self.x_torch = torch.tensor(self.x_np, requires_grad=True)

        # Create the Conv2D layers
        self.conv_custom = Conv2D(num_filters=3, kernel_size=(3, 3), padding="same")
        self.conv_torch = Conv2d(in_channels=self.x_np.shape[3], out_channels=3, kernel_size=3, padding=1)

        # Initialize the custom layer
        self.conv_custom.eval()
        self.conv_custom(self.x_tensor)

        # Copy the weights and bias from the custom layer to the PyTorch layer
        with torch.no_grad():
            # Extract the weights and bias from the custom layer
            w_custom = torch.from_numpy(self.conv_custom.filters.data).float()
            b_custom = torch.from_numpy(self.conv_custom.bias.data).float()
            
            # Copy the weights and bias to the PyTorch layer
            self.conv_torch.weight.copy_(w_custom)
            if self.conv_torch.bias is not None: self.conv_torch.bias.copy_(b_custom) 


    def test_conv_2d_forward(self) -> None:
        """
        Test to verify that the forward pass of the Conv2D layer is consistent with PyTorch.
        """
        
        # Forward pass
        y_custom = self.conv_custom(self.x_tensor)
        y_torch = self.conv_torch(self.x_torch.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Compare the forward pass results
        self.assertTrue(
            np.allclose(y_custom.data, y_torch.detach().numpy(), atol=1e-5),
            msg=(
                f"❌ Forward outputs differ beyond tolerance!\n"
                f"Custom: {y_custom.data}\n"
                f"Torch: {y_torch.detach().numpy()}"
            )
        )


    def test_conv_2d_backward(self) -> None:
        """
        Test to verify that the backward pass of the Conv2D layer is consistent with PyTorch.
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
                f"Custom grad:\n{self.x_tensor.grad}\n\n"
                f"Torch grad:\n{self.x_torch.grad.numpy()}"
            )
        )


if __name__ == "__main__":
    unittest.main()