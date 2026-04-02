import os
import sys
import torch
import unittest
import numpy as np
from torch.nn import LSTM as TorchLSTM

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.core import Tensor
from src.tests.base import Test
from src.layers import LSTM as CustomLSTM


class TestLSTMLayer(Test):

    def setUp(self) -> None:
        """
        Set up the test case for LSTM.
        This method will be called before each test.
        """

        # Create random input data
        self.x_np = np.random.randn(4, 8, 3).astype(np.float32)
        
        # Create the input tensors
        self.x_tensor = Tensor(self.x_np, requires_grad=True)
        self.x_torch = torch.tensor(self.x_np, requires_grad=True)

        # Create the custom LSTM layer
        self.layer_custom = CustomLSTM(
            num_layers = 8,
            num_units = 32,
            add_bias = True,
            return_sequences = True
        )
        
        # Create the PyTorch LSTM layer
        self.layer_torch = TorchLSTM(
            input_size = self.x_np.shape[2],
            hidden_size = 32,
            num_layers = 8,
            bias = True,
            batch_first = True
        )

        # Initialize the custom layer (to build weights)
        self.layer_custom.eval()
        self.layer_custom(self.x_tensor)
               
        # Copy weights and biases from custom layer to PyTorch layer     
        with torch.no_grad():
            # Iterate through each layer of the custom LSTM
            for i in range(self.layer_custom.num_layers):
                # Extract weights
                w_ih_numpy = self.layer_custom.W[i].data
                w_hh_numpy = self.layer_custom.U[i].data

                # Transpose to match PyTorch layout
                w_ih = torch.from_numpy(w_ih_numpy).float().t()
                w_hh = torch.from_numpy(w_hh_numpy).float().t()

                # Copy to PyTorch parameters
                getattr(self.layer_torch, f'weight_ih_l{i}').data.copy_(w_ih)
                getattr(self.layer_torch, f'weight_hh_l{i}').data.copy_(w_hh)

                # Check if add_bias is enabled
                if self.layer_custom.add_bias:
                    # Extract custom biases
                    b_ih_numpy = self.layer_custom.bias_ih[i].data
                    b_hh_numpy = self.layer_custom.bias_hh[i].data

                    # Convert to PyTorch tensors
                    b_ih = torch.from_numpy(b_ih_numpy).float()
                    b_hh = torch.from_numpy(b_hh_numpy).float()

                    # Copy to PyTorch parameters
                    getattr(self.layer_torch, f'bias_ih_l{i}').data.copy_(b_ih)
                    getattr(self.layer_torch, f'bias_hh_l{i}').data.copy_(b_hh)


    def test_LSTM_forward(self) -> None:
        """
        Test to verify that the forward pass of the LSTM layer matches PyTorch.
        """
        
        # Forward pass
        y_custom = self.layer_custom(self.x_tensor).output
        y_torch, _ = self.layer_torch(self.x_torch)

        self.assertTrue(
            np.allclose(y_custom.data, y_torch.detach().numpy(), atol=1e-5),
            msg=(
                f"❌ Forward outputs differ beyond tolerance!\n"
                f"Custom: {y_custom.data}\n"
                f"Torch: {y_torch.detach().numpy()}"
            )
        )


    def test_LSTM_backward(self) -> None:
        """
        Test to verify that the backward pass of the LSTM layer matches PyTorch.
        """
        
        # Forward pass
        y_custom = self.layer_custom(self.x_tensor).output
        y_torch, _ = self.layer_torch(self.x_torch)

        # Define loss and backpropagate
        loss_custom = y_custom.sum()
        loss_torch = y_torch.sum()
        loss_custom.backward()
        loss_torch.backward()

        # Assert gradients exist
        self.assertIsNotNone(self.x_tensor.grad, "Custom Tensor grad is None")
        self.assertIsNotNone(self.x_torch.grad, "PyTorch Tensor grad is None")
        
        # Check if the gradients are not None
        if self.x_tensor.grad is None or self.x_torch.grad is None:
            self.fail("Gradients are None!")

        # Compare gradients
        self.assertTrue(
            np.allclose(self.x_tensor.grad, self.x_torch.grad.numpy(), atol=1e-5),
            msg=(
                f"❌ Backward gradients differ beyond tolerance!\n"
                f"Custom grad:\n{self.x_tensor.grad}\n"
                f"Torch grad:\n{self.x_torch.grad.numpy()}"
            )
        )


if __name__ == "__main__":
    unittest.main()
