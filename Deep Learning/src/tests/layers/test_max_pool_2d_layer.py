import os
import sys
import torch
import unittest
import numpy as np
from torch.nn import MaxPool2d

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.core import Tensor
from src.tests.base import Test
from src.layers import MaxPool2D


class TestMaxPool2DLayer(Test):

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

        # Create the MaxPool2D layers
        self.max_pool_custom = MaxPool2D(size=(2,2), stride=(2,2))
        self.max_pool_torch = MaxPool2d(kernel_size=2, stride=2)

        # Initialize the custom layer
        self.max_pool_custom.eval()
        self.max_pool_custom(self.x_tensor)


    def test_max_pool_2d_forward(self) -> None:
        """
        Test to verify that the forward pass of the MaxPool2D layer is consistent with PyTorch.
        """
        
        # Forward pass
        y_custom = self.max_pool_custom(self.x_tensor)
        y_torch = self.max_pool_torch(self.x_torch.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Compare the forward pass results
        self.assertTrue(
            np.allclose(y_custom.data, y_torch.detach().numpy(), atol=1e-5),
            msg=(
                f"❌ Forward outputs differ beyond tolerance!\n"
                f"Custom: {y_custom.data}\n"
                f"Torch: {y_torch.detach().numpy()}"
            )
        )


    def test_max_pool_2d_backward(self) -> None:
        """
        Test to verify that the backward pass of the MaxPool2D layer is consistent with PyTorch.
        """
        
        # Forward pass
        y_custom = self.max_pool_custom(self.x_tensor)
        y_torch = self.max_pool_torch(self.x_torch.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
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