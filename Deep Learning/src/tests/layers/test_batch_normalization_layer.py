import os
import sys
import torch
import unittest
import numpy as np
from torch.nn import BatchNorm1d

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.core import Tensor
from src.tests.base import Test
from src.layers import BatchNormalization


class TestBatchNormalizationLayer(Test):

    def setUp(self) -> None:
        """
        Set up the test case.
        This method will be called before each test.
        """

        # Create random input data
        self.x_np = np.random.randn(3, 8).astype(np.float32)
        
        # Create the input tensors
        self.x_tensor = Tensor(self.x_np, requires_grad=True)
        self.x_torch = torch.tensor(self.x_np, requires_grad=True)

        # Create the BatchNormalization layers
        self.layer_custom = BatchNormalization()
        self.layer_torch = BatchNorm1d(num_features=self.x_np.shape[-1])

        # Initialize the custom layer
        self.layer_custom.eval()
        self.layer_custom(self.x_tensor)

        # Copy the parameters from the custom layer to the PyTorch layer
        with torch.no_grad():
            # Copy gamma to weight and beta to bias
            gamma_custom = torch.from_numpy(self.layer_custom.gamma.data).float()
            beta_custom = torch.from_numpy(self.layer_custom.beta.data).float()
            self.layer_torch.weight.copy_(gamma_custom)
            if self.layer_torch.bias is not None:
                self.layer_torch.bias.copy_(beta_custom)

            # Copy the running statistics
            running_mean_custom = torch.from_numpy(self.layer_custom.running_mean.data).float()
            running_var_custom = torch.from_numpy(self.layer_custom.running_var.data).float()
            
            # Copy the running statistics
            if self.layer_torch.running_mean is not None:
                self.layer_torch.running_mean.copy_(running_mean_custom)
            if self.layer_torch.running_var is not None:
                self.layer_torch.running_var.copy_(running_var_custom)


    def test_batch_norm_forward_train(self) -> None:
        """
        Test to verify that the forward pass of the BatchNormalization layer in training mode is consistent with PyTorch.
        """
        
        # Set the layers to training mode
        self.layer_custom.train()
        self.layer_torch.train()
        
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


    def test_batch_norm_forward_eval(self) -> None:
        """
        Test to verify that the forward pass of the BatchNormalization in evaluation mode layer is consistent with PyTorch.
        """
        
        # Set the layers to evaluation mode
        self.layer_custom.eval()
        self.layer_torch.eval()
        
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


    def test_batch_norm_backward(self) -> None:
        """
        Test to verify that the backward pass of the BatchNormalization layer is consistent with PyTorch.
        """
        
        # Set the layers to training mode
        self.layer_custom.train()
        self.layer_torch.train()
        
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