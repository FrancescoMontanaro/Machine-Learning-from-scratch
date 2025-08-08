import os
import sys
import torch
import unittest
import numpy as np
from torch.nn import LocalResponseNorm as TorchLocalResponseNorm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.core import Tensor
from src.tests.base import Test
from src.layers import LocalResponseNormalization


class TestLocalResponseNormLayer(Test):

    def setUp(self) -> None:
        """
        Set up the test case.
        This method will be called before each test.
        """

        # Create random input data (batch_size, height, width, channels)
        self.x_np = np.random.randn(2, 4, 4, 3).astype(np.float32)
        
        # Create the input tensors
        self.x_tensor = Tensor(self.x_np, requires_grad=True)
        
        # PyTorch expects (batch, channels, height, width) format
        self.x_torch = torch.tensor(
            self.x_np.transpose(0, 3, 1, 2),
            requires_grad=True
        )

        # LRN parameters
        self.size = 5
        self.alpha = 1e-4
        self.beta = 0.75
        self.k = 1.0

        # Create the LocalResponseNorm layers
        self.layer_custom = LocalResponseNormalization(
            size = self.size,
            alpha = self.alpha,
            beta = self.beta,
            k = self.k
        )
        
        self.layer_torch = TorchLocalResponseNorm(
            size = self.size,
            alpha = self.alpha,
            beta = self.beta,
            k = self.k
        )

        # Initialize the custom layer
        self.layer_custom(self.x_tensor)


    def test_lrn_forward(self) -> None:
        """
        Test to verify that the forward pass of the LocalResponseNorm layer is consistent with PyTorch.
        """
        
        # Forward pass
        y_custom = self.layer_custom(self.x_tensor)
        y_torch = self.layer_torch(self.x_torch)

        # Convert PyTorch output back to (batch, height, width, channels) format
        y_torch_transposed = y_torch.detach().numpy().transpose(0, 2, 3, 1)

        # Compare the forward pass results
        self.assertTrue(
            np.allclose(y_custom.data, y_torch_transposed, atol=1e-5),
            msg=(
                f"❌ Forward outputs differ beyond tolerance!\n"
                f"Custom shape: {y_custom.data.shape}\n"
                f"PyTorch shape: {y_torch_transposed.shape}\n"
                f"Max diff: {np.max(np.abs(y_custom.data - y_torch_transposed))}"
            )
        )


    def test_lrn_backward(self) -> None:
        """
        Test to verify that the backward pass of the LocalResponseNorm layer is consistent with PyTorch.
        """
        
        # Forward pass
        y_custom = self.layer_custom(self.x_tensor)
        y_torch = self.layer_torch(self.x_torch)
        
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

        # Convert PyTorch gradient back to (batch, height, width, channels) format
        torch_grad_transposed = self.x_torch.grad.numpy().transpose(0, 2, 3, 1)

        # Compare the backward gradients
        self.assertTrue(
            np.allclose(self.x_tensor.grad, torch_grad_transposed, atol=1e-4),
            msg=(
                f"❌ Backward gradients differ beyond tolerance!\n"
                f"Custom grad shape: {self.x_tensor.grad.shape}\n"
                f"PyTorch grad shape: {torch_grad_transposed.shape}\n"
                f"Max diff: {np.max(np.abs(self.x_tensor.grad - torch_grad_transposed))}"
            )
        )
        
        
if __name__ == "__main__":
    unittest.main()