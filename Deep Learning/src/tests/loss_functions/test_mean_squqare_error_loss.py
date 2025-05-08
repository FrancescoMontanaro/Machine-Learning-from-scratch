import os
import sys
import torch
import unittest
import numpy as np
from torch.nn import MSELoss

# Add the project root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.core import Tensor
from src.tests.base import Test
from src.loss_functions import MeanSquareError


class TestMSELoss(Test):
    
    def setUp(self) -> None:
        """
        Set up the test case.
        This method will be called before each test.
        """

        # Create random prediction and target arrays
        self.y_pred_np = np.random.randn(4, 5).astype(np.float32)
        self.y_target_np = np.random.randn(4, 5).astype(np.float32)

        # Create custom Tensors for predictions and targets
        self.y_pred_tensor = Tensor(self.y_pred_np, requires_grad=True)
        self.y_target_tensor = Tensor(self.y_target_np)

        # Create PyTorch tensors for predictions and targets
        self.y_pred_torch = torch.tensor(self.y_pred_np, requires_grad=True)
        self.y_target_torch = torch.tensor(self.y_target_np)

        # Instantiate the loss functions
        self.loss_custom = MeanSquareError()
        self.loss_torch = MSELoss()


    def test_mse_loss_forward(self) -> None:
        """
        Test to verify that the forward pass of the custom MSELoss 
        is consistent with PyTorch's MSELoss.
        """
        
        # Compute the loss values for custom and torch implementations
        loss_custom_val = self.loss_custom(self.y_pred_tensor, self.y_target_tensor)
        loss_torch_val = self.loss_torch(self.y_pred_torch, self.y_target_torch)

        # Compare the forward loss values
        self.assertTrue(
            np.allclose(loss_custom_val.data, loss_torch_val.item(), atol=1e-5),
            msg=(
                f"❌ Forward loss outputs differ!\n"
                f"Custom Loss: {loss_custom_val.data}\n"
                f"Torch Loss: {loss_torch_val.item()}"
            )
        )


    def test_mse_loss_backward(self) -> None:
        """
        Test to verify that the backward pass (gradient computation) 
        of the custom MSELoss is consistent with PyTorch's MSELoss.
        """
        
        # Compute the losses
        loss_custom_val = self.loss_custom(self.y_pred_tensor, self.y_target_tensor)
        loss_torch_val = self.loss_torch(self.y_pred_torch, self.y_target_torch)

        # Perform the backward pass on both losses
        loss_custom_val.backward()
        loss_torch_val.backward()

        # Check that gradients are not None
        self.assertIsNotNone(self.y_pred_tensor.grad, "Custom Tensor grad is None")
        self.assertIsNotNone(self.y_pred_torch.grad, "PyTorch Tensor grad is None")
        
        # Check if the gradients are not None
        if self.y_pred_tensor.grad is None or self.y_pred_torch.grad is None:
            self.fail("Gradients are None!")

        # Compare the gradients for the predictions
        self.assertTrue(
            np.allclose(self.y_pred_tensor.grad, self.y_pred_torch.grad.numpy(), atol=1e-5),
            msg=(
                f"❌ Backward gradients differ!\n"
                f"Custom grad:\n{self.y_pred_tensor.grad}\n\n"
                f"Torch grad:\n{self.y_pred_torch.grad.numpy()}"
            )
        )


if __name__ == "__main__":
    unittest.main()