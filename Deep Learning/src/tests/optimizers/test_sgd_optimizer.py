import os
import sys
import torch
import unittest
import numpy as np
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.core import Tensor
from src.layers import Dense
from src.tests.base import Test
from src.optimizers import SGD as CustomSGD


class TestSGDOptimizer(Test):

    def setUp(self) -> None:
        """
        Set up the test case.
        This method will be called before each test.
        """

        # Create random input data
        self.x_np = np.random.randn(4, 8).astype(np.float32)
        self.y_np = np.random.randn(4, 1).astype(np.float32)
        
        # Create the input tensors
        self.x_tensor = Tensor(self.x_np, requires_grad=True)
        self.y_tensor = Tensor(self.y_np, requires_grad=False)
        self.x_torch = torch.tensor(self.x_np, requires_grad=True)
        self.y_torch = torch.tensor(self.y_np, requires_grad=False)

        # Create the models
        self.model_custom = Dense(num_units=1)
        self.model_torch = nn.Linear(in_features=self.x_np.shape[1], out_features=1)

        # Initialize the custom model
        self.model_custom.eval()
        self.model_custom(self.x_tensor)

        # Copy the weights and bias from the custom model to the PyTorch model
        with torch.no_grad():
            # Extract the weights and bias from the custom model
            w_custom = torch.from_numpy(self.model_custom.weights.data.transpose()).float()
            b_custom = torch.from_numpy(self.model_custom.bias.data.transpose()).float()
            
            # Copy the weights and bias to the PyTorch model
            self.model_torch.weight.copy_(w_custom)
            if self.model_torch.bias is not None: 
                self.model_torch.bias.copy_(b_custom)

        # Learning rate and momentum for testing
        self.lr = 0.01
        self.momentum = 0.9

        # Create the custom optimizers
        self.optimizer_custom = CustomSGD(
            learning_rate = self.lr,
            momentum = self.momentum,
            parameters = [self.model_custom.weights, self.model_custom.bias]
        )
        
        # Create the PyTorch optimizer
        self.optimizer_torch = torch.optim.SGD(
            params = self.model_torch.parameters(),
            lr = self.lr,
            momentum = self.momentum
        )
        

    def test_sgd_optimizer_step(self) -> None:
        """
        Test that a single optimization step produces similar parameter updates
        in both custom and PyTorch implementations.
        """
        
        # Forward pass - Custom
        y_pred_custom = self.model_custom(self.x_tensor)
        loss_custom = ((y_pred_custom - self.y_tensor) ** 2).mean()
        
        # Forward pass - PyTorch
        y_pred_torch = self.model_torch(self.x_torch)
        loss_torch = ((y_pred_torch - self.y_torch) ** 2).mean()

        # Backward pass - Custom
        loss_custom.backward()
        
        # Backward pass - PyTorch
        loss_torch.backward()

        # Get parameters before step
        w_custom_before = self.model_custom.weights.data.copy()
        b_custom_before = self.model_custom.bias.data.copy()
        w_torch_before = self.model_torch.weight.data.clone().numpy().T
        b_torch_before = self.model_torch.bias.data.clone().numpy().T

        # Optimization step
        self.optimizer_custom.update()
        self.optimizer_torch.step()

        # Get parameters after step
        w_custom_after = self.model_custom.weights.data
        b_custom_after = self.model_custom.bias.data
        w_torch_after = self.model_torch.weight.data.clone().numpy().T
        b_torch_after = self.model_torch.bias.data.clone().numpy().T

        # Calculate parameter changes
        w_custom_change = w_custom_after - w_custom_before
        b_custom_change = b_custom_after - b_custom_before
        w_torch_change = w_torch_after - w_torch_before
        b_torch_change = b_torch_after - b_torch_before

        # Compare parameter changes
        self.assertTrue(
            np.allclose(w_custom_change, w_torch_change, atol=1e-5),
            msg=(
                f"❌ Weight updates differ beyond tolerance!\n"
                f"Custom weight change:\n{w_custom_change}\n\n"
                f"Torch weight change:\n{w_torch_change}"
            )
        )

        self.assertTrue(
            np.allclose(b_custom_change, b_torch_change, atol=1e-5),
            msg=(
                f"❌ Bias updates differ beyond tolerance!\n"
                f"Custom bias change:\n{b_custom_change}\n\n"
                f"Torch bias change:\n{b_torch_change}"
            )
        )


    def test_sgd_optimizer_zero_grad(self) -> None:
        """
        Test that zero_grad() properly clears the gradients.
        """
        
        # Forward pass - Custom
        y_pred_custom = self.model_custom(self.x_tensor)
        loss_custom = ((y_pred_custom - self.y_tensor) ** 2).mean()
        
        # Forward pass - PyTorch
        y_pred_torch = self.model_torch(self.x_torch)
        loss_torch = ((y_pred_torch - self.y_torch) ** 2).mean()

        # Backward pass - Custom
        loss_custom.backward()
        
        # Backward pass - PyTorch
        loss_torch.backward()

        # Verify gradients exist before zero_grad
        self.assertIsNotNone(self.model_custom.weights.grad, "Custom weights grad is None before zero_grad")
        self.assertIsNotNone(self.model_custom.bias.grad, "Custom bias grad is None before zero_grad")
        self.assertIsNotNone(self.model_torch.weight.grad, "Torch weight grad is None before zero_grad")
        self.assertIsNotNone(self.model_torch.bias.grad, "Torch bias grad is None before zero_grad")

        # Zero gradients
        self.optimizer_custom.zero_grad()
        self.optimizer_torch.zero_grad()
        
        # Verify gradients are None after zero_grad
        self.assertIsNone(self.model_custom.weights.grad, "Custom weights grad is not None after zero_grad")
        self.assertIsNone(self.model_custom.bias.grad, "Custom bias grad is not None after zero_grad")
        self.assertIsNone(self.model_torch.weight.grad, "Torch weight grad is not None after zero_grad")
        self.assertIsNone(self.model_torch.bias.grad, "Torch bias grad is not None after zero_grad")