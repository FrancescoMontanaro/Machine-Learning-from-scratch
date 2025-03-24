import os
import sys
import torch
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core import Tensor
from src.layers import Reshape


class TestReshape(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up the test case.
        This method will be called before each test.
        """
        
        # Set the seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # Create random input data
        batch_size = 4
        self.x_np = np.random.randn(batch_size, 8).astype(np.float32)
        
        # Create the input tensors
        self.x_tensor = Tensor(self.x_np, requires_grad=True)
        self.x_torch = torch.tensor(self.x_np, requires_grad=True)

        # Create the layers
        self.layer_custom = Reshape(shape=(4, 2))
        self.layer_torch = lambda x: x.reshape(batch_size, 4, 2)


    def test_reshape_forward(self) -> None:
        """
        Test to verify that the forward pass of the Reshape layer is consistent with PyTorch.
        """
        
        # Forward pass
        y_custom = self.layer_custom(self.x_tensor)
        y_torch = self.layer_torch(self.x_torch)

        # Compare the forward pass results
        self.assertTrue(
            np.allclose(y_custom.data, y_torch.detach().numpy(), atol=1e-5),
            msg=(
                f"❌ Forward outputs differ beyond tolerance!\n"
                f"Custom: {y_custom.data}\n"
                f"Torch: {y_torch.detach().numpy()}"
            )
        )


    def test_reshape_backward(self) -> None:
        """
        Test to verify that the backward pass of the Reshape layer is consistent with PyTorch.
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
            raise AssertionError("Gradients are None!")

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