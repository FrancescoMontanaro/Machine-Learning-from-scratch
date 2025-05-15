import os
import sys
import torch
import unittest
import numpy as np
from torch.nn import RNN as TorchRNN

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.core import Tensor
from src.tests.base import Test
from src.activations import Tanh
from src.layers import RNN as CustomRNN


class TestRNNLayer(Test):

    def setUp(self) -> None:
        """
        Set up the test case.
        This method will be called before each test.
        """

        # Create random input data
        self.x_np = np.random.randn(8, 24, 3).astype(np.float32)
        
        # Create the input tensors
        self.x_tensor = Tensor(self.x_np, requires_grad=True)
        self.x_torch = torch.tensor(self.x_np, requires_grad=True)

        # Create the costum layer
        self.layer_custom = CustomRNN(
            num_layers = 8, 
            num_units = 32, 
            activation = Tanh(), 
            add_bias = True,
            return_sequences = True
        )
        
        # Create the PyTorch layer
        self.layer_torch = TorchRNN(
            input_size = self.x_np.shape[2],
            hidden_size = 32,
            num_layers = 8,
            bias = True,
            nonlinearity = 'tanh',
            batch_first = True
        )

        # Initialize the custom layer
        self.layer_custom.eval()
        self.layer_custom(self.x_tensor)
               
        # Copy weights and biases from custom layer to PyTorch layer     
        with torch.no_grad():
            # Iterate through each layer of the custom RNN
            for i in range(self.layer_custom.num_layers):
                # Extract the numpy weight and bias arrays from the custom layer
                w_ih_numpy = self.layer_custom.weights[i].data
                w_hh_numpy = self.layer_custom.recurrent_weights[i].data
                
                # Convert to PyTorch tensors and transpose
                w_ih_custom_transposed = torch.from_numpy(w_ih_numpy).float().t()
                w_hh_custom_transposed = torch.from_numpy(w_hh_numpy).float().t()

                # Get target PyTorch parameters
                target_w_ih = getattr(self.layer_torch, f'weight_ih_l{i}')
                target_w_hh = getattr(self.layer_torch, f'weight_hh_l{i}')
                
                # Copy the weights from custom to PyTorch
                target_w_ih.data.copy_(w_ih_custom_transposed)
                target_w_hh.data.copy_(w_hh_custom_transposed)

                # Check if the layer has biases
                if self.layer_custom.add_bias:
                    # Extract the numpy bias arrays from the custom layer
                    b_ih_numpy = self.layer_custom.bias[i].data
                    b_hh_numpy = self.layer_custom.recurrent_bias[i].data
                    
                    # Convert to PyTorch tensors
                    b_ih_custom = torch.from_numpy(b_ih_numpy).float()
                    b_hh_custom = torch.from_numpy(b_hh_numpy).float()
                    
                    # Get target PyTorch biases
                    target_b_ih = getattr(self.layer_torch, f'bias_ih_l{i}')
                    target_b_hh = getattr(self.layer_torch, f'bias_hh_l{i}')

                    # Copy the biases from custom to PyTorch
                    target_b_ih.data.copy_(b_ih_custom)
                    target_b_hh.data.copy_(b_hh_custom)


    def test_RNN_forward(self) -> None:
        """
        Test to verify that the forward pass of the RNN layer is consistent with PyTorch.
        """
        
        # Forward pass
        y_custom = self.layer_custom(self.x_tensor)
        y_torch, _ = self.layer_torch(self.x_torch)

        # Compare the forward pass results
        self.assertTrue(
            np.allclose(y_custom.data, y_torch.detach().numpy(), atol=1e-5),
            msg=(
                f"❌ Forward outputs differ beyond tolerance!\n"
                f"Custom: {y_custom.data}\n"
                f"Torch: {y_torch.detach().numpy()}"
            )
        )


    def test_RNN_backward(self) -> None:
        """
        Test to verify that the backward pass of the RNN layer is consistent with PyTorch.
        """
        
        # Forward pass
        y_custom = self.layer_custom(self.x_tensor)
        y_torch, _ = self.layer_torch(self.x_torch)
        
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