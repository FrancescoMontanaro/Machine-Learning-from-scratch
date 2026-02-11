import os
import sys
import torch
import unittest
import numpy as np
from torch.nn import KLDivLoss

# Add the project root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.core import Tensor
from src.tests.base import Test
from src.loss_functions import KullbackLeiblerDivergence


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax using NumPy.
    """

    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


class TestKLDivergenceLoss(Test):

    def setUp(self) -> None:
        """
        Set up the test case.
        This method will be called before each test.
        """

        # Create random logits for predictions and targets
        self.y_pred_np = np.random.randn(4, 5).astype(np.float32)
        self.y_target_logits_np = np.random.randn(4, 5).astype(np.float32)

        # Convert target logits to probability distribution
        self.y_target_probs_np = _softmax_np(self.y_target_logits_np, axis=-1).astype(np.float32)

        # Create custom Tensors for predictions and targets
        self.y_pred_tensor = Tensor(self.y_pred_np, requires_grad=True)
        self.y_target_tensor = Tensor(self.y_target_probs_np)

        # Create PyTorch tensors for predictions and targets
        self.y_pred_torch = torch.tensor(self.y_pred_np, requires_grad=True)
        self.y_target_torch = torch.tensor(self.y_target_probs_np)

        # Instantiate the loss functions
        self.loss_custom = KullbackLeiblerDivergence()
        self.loss_torch = KLDivLoss(reduction="batchmean")


    def test_kld_loss_forward(self) -> None:
        """
        Test to verify that the forward pass of the custom KLDivergenceLoss
        is consistent with PyTorch's KLDivLoss.
        """

        # Compute the loss values for custom and torch implementations
        loss_custom_val = self.loss_custom(self.y_target_tensor, self.y_pred_tensor).output
        loss_torch_val = self.loss_torch(
            torch.log_softmax(self.y_pred_torch, dim=-1),
            self.y_target_torch
        )

        # Compare the forward loss values
        self.assertTrue(
            np.allclose(loss_custom_val.data, loss_torch_val.item(), atol=1e-5),
            msg=(
                f"❌ Forward loss outputs differ!\n"
                f"Custom Loss: {loss_custom_val.data}\n"
                f"Torch Loss: {loss_torch_val.item()}"
            )
        )


    def test_kld_loss_backward(self) -> None:
        """
        Test to verify that the backward pass (gradient computation)
        of the custom KLDivergenceLoss is consistent with PyTorch's KLDivLoss.
        """

        # Compute the losses
        loss_custom_val = self.loss_custom(self.y_target_tensor, self.y_pred_tensor).output
        loss_torch_val = self.loss_torch(
            torch.log_softmax(self.y_pred_torch, dim=-1),
            self.y_target_torch
        )

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
