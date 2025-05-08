import os
import sys
import torch
import unittest
import numpy as np
from torch.nn import Embedding as TorchEmbedding

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.core import Tensor
from src.tests.base import Test
from src.layers import Embedding


class TestEmbeddingLayer(Test):

    def setUp(self) -> None:
        """
        Set up the test case.
        This method will be called before each test.
        """

        # Define vocabulary size, embedding dimension and batch size
        self.vocab_size = 10
        self.embedding_dim = 3
        self.batch_size = 4

        # Create random indices as input data with shape (batch_size, 1)
        self.x_indices_np = np.random.randint(0, self.vocab_size, (self.batch_size, 1))
        
        # Create the input tensors
        self.x_tensor = Tensor(self.x_indices_np, requires_grad=True)
        self.x_torch = torch.tensor(self.x_indices_np, dtype=torch.long)

        # Create the layers
        self.layer_custom = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)
        self.layer_torch = TorchEmbedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)

        # Initialize the custom layer's parameters
        self.layer_custom.eval()
        self.layer_custom(self.x_tensor)

        # Copy the parameters from the custom layer to the PyTorch layer for a fair comparison.
        with torch.no_grad():
            # Convert the custom layer's embedding matrix to a PyTorch tensor and copy it to the PyTorch layer
            custom_weight = torch.from_numpy(self.layer_custom.embedding.data).float()
            self.layer_torch.weight.copy_(custom_weight)
            

    def test_embedding_forward(self) -> None:
        """
        Test to verify that the forward pass of the Embedding layer is consistent with PyTorch.
        """
        
        # Forward pass through both layers
        y_custom = self.layer_custom(self.x_tensor)
        y_torch = self.layer_torch(self.x_torch)

        # Compare the outputs: both should be of shape (batch_size, embedding_dim)
        self.assertTrue(
            np.allclose(y_custom.data, y_torch.detach().numpy(), atol=1e-5),
            msg=(
                f"❌ Forward outputs differ beyond tolerance!\n"
                f"Custom output:\n{y_custom.data}\n"
                f"PyTorch output:\n{y_torch.detach().numpy()}"
            )
        )
        

    def test_embedding_backward(self) -> None:
        """
        Test to verify that the backward pass of the Embedding layer is consistent with PyTorch.
        """
        
        # Forward pass
        y_custom = self.layer_custom(self.x_tensor)
        y_torch = self.layer_torch(self.x_torch)

        # Define a simple loss as the sum of all outputs and perform backward pass
        loss_custom = y_custom.sum()
        loss_torch = y_torch.sum()
        loss_custom.backward()
        loss_torch.backward()

        # Ensure that gradients exist for the embedding matrix
        self.assertIsNotNone(self.layer_custom.embedding.grad, "Custom embedding grad is None")
        self.assertIsNotNone(self.layer_torch.weight.grad, "PyTorch embedding weight grad is None")
        
        # Check if the gradients are not None
        if self.layer_custom.embedding.grad is None or self.layer_torch.weight.grad is None:
            self.fail("Gradients are None!")

        # Compare the gradients of the embedding matrix
        self.assertTrue(
            np.allclose(self.layer_custom.embedding.grad, self.layer_torch.weight.grad.detach().numpy(), atol=1e-5),
            msg=(
                f"❌ Backward gradients differ beyond tolerance!\n"
                f"Custom embedding grad:\n{self.layer_custom.embedding.grad}\n\n"
                f"PyTorch embedding grad:\n{self.layer_torch.weight.grad.detach().numpy()}"
            )
        )


if __name__ == "__main__":
    unittest.main()