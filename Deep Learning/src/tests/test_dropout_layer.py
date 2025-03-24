import os
import sys
import torch
import unittest
import numpy as np
from torch.nn import Dropout as TorchDropout

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core import Tensor
from src.layers import Dropout as CustomDropout


class TestDropout(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up the test case.
        This method will be called before each test.
        """
        
        # Set the seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # Create random input data
        self.x_np = np.random.randn(3, 8).astype(np.float32)
        
        # Create the input tensors
        self.x_tensor = Tensor(self.x_np, requires_grad=True)
        self.x_torch = torch.tensor(self.x_np, requires_grad=True)

        # Create the Dropout layers
        self.layer_custom = CustomDropout(rate=0.5)
        self.layer_torch = TorchDropout(p=0.5)


    def test_dropout_forward_eval(self) -> None:
        """
        Test to verify that the forward pass of the Dropout in evaluation mode layer is consistent with PyTorch.
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


if __name__ == "__main__":
    unittest.main()