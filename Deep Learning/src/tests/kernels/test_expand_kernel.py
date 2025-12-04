import os
import sys
import time
import torch
import unittest
import numpy as np

# Ensure the src directory is on sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src import Tensor
from src.tests.base import Test


class TestExpandKernel(Test):
    
    def setUp(self):
        """
        Set up the test case.
        """
        
        # Seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)


    def test_expand_forward_simple(self):
        """
        Test that expand matches torch.expand on a simple case.
        Expand a (1, 4) tensor to (3, 4).
        """
        
        # Create random input data
        x_raw = np.random.rand(1, 4).astype(np.float32)
        
        # Custom tensor
        x_tensor = Tensor(x_raw, dtype=np.float32, requires_grad=False)

        # Torch tensor
        x_torch = torch.tensor(x_raw, dtype=torch.float32, requires_grad=False)
        
        # PyTorch expand
        out_torch = x_torch.expand(3, 4).numpy()

        # Custom expand
        out_custom = x_tensor.expand(3, 4).detach().to_numpy()

        # Assert values are close
        self.assertTrue(
            np.allclose(out_custom, out_torch, atol=1e-6),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom shape: {out_custom.shape}, Torch shape: {out_torch.shape}\n"
                f"Custom: {out_custom}\n"
                f"Torch: {out_torch}"
            )
        )
        

    def test_expand_forward_with_minus_one(self):
        """
        Test that expand works with -1 (keep original dimension).
        Expand a (2, 1, 4) tensor to (2, 3, 4).
        """
        
        # Create random input data
        x_raw = np.random.rand(2, 1, 4).astype(np.float32)
        
        # Custom tensor
        x_tensor = Tensor(x_raw, dtype=np.float32, requires_grad=False)

        # Torch tensor
        x_torch = torch.tensor(x_raw, dtype=torch.float32, requires_grad=False)
        
        # PyTorch expand with -1
        out_torch = x_torch.expand(-1, 3, -1).numpy()

        # Custom expand with -1
        out_custom = x_tensor.expand(-1, 3, -1).detach().to_numpy()

        # Assert values are close
        self.assertTrue(
            np.allclose(out_custom, out_torch, atol=1e-6),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom shape: {out_custom.shape}, Torch shape: {out_torch.shape}\n"
                f"Custom: {out_custom}\n"
                f"Torch: {out_torch}"
            )
        )


    def test_expand_forward_add_dimension(self):
        """
        Test that expand can add new dimensions at the front.
        Expand a (3, 1) tensor to (2, 3, 4).
        """
        
        # Create random input data
        x_raw = np.random.rand(3, 1).astype(np.float32)
        
        # Custom tensor
        x_tensor = Tensor(x_raw, dtype=np.float32, requires_grad=False)

        # Torch tensor
        x_torch = torch.tensor(x_raw, dtype=torch.float32, requires_grad=False)
        
        # PyTorch expand (adds batch dimension)
        out_torch = x_torch.expand(2, 3, 4).numpy()

        # Custom expand
        out_custom = x_tensor.expand(2, 3, 4).detach().to_numpy()

        # Assert values are close
        self.assertTrue(
            np.allclose(out_custom, out_torch, atol=1e-6),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom shape: {out_custom.shape}, Torch shape: {out_torch.shape}\n"
                f"Custom: {out_custom}\n"
                f"Torch: {out_torch}"
            )
        )


    def test_expand_backward_simple(self):
        """
        Test that expand backward matches torch.expand backward gradients.
        """
        
        # Create random input data
        x_raw = np.random.rand(1, 4).astype(np.float32)
        
        # Setup tensors with grad
        x_torch = torch.tensor(x_raw, dtype=torch.float32, requires_grad=True)
        
        # Custom tensor
        x_custom = Tensor(x_raw, dtype=np.float32, requires_grad=True)

        # PyTorch forward + backward
        z_torch = x_torch.expand(3, 4)
        z_torch.sum().backward()
        
        # Check gradients
        assert x_torch.grad is not None, "Gradient for x is None"

        # Detach gradients to numpy
        grad_x_torch = x_torch.grad.detach().numpy()

        # Custom forward + backward
        z_custom = x_custom.expand(3, 4)
        z_custom.sum().backward()
        
        # Check gradients
        assert x_custom.grad is not None, "Gradient for x is None"
        
        # Detach gradients to numpy
        grad_x_custom = x_custom.grad

        # Assert gradients are equal
        self.assertTrue(
            np.allclose(grad_x_custom, grad_x_torch, atol=1e-6),
            msg=(
                f"❌ Backward gradients differ!\n"
                f"Custom grad shape: {grad_x_custom.shape}, Torch grad shape: {grad_x_torch.shape}\n"
                f"Custom grad: {grad_x_custom}\n"
                f"Torch grad: {grad_x_torch}"
            )
        )


    def test_expand_backward_with_minus_one(self):
        """
        Test that expand backward works correctly with -1 dimensions.
        """
        
        # Create random input data
        x_raw = np.random.rand(2, 1, 4).astype(np.float32)
        
        # Setup tensors with grad
        x_torch = torch.tensor(x_raw, dtype=torch.float32, requires_grad=True)
        
        # Custom tensor
        x_custom = Tensor(x_raw, dtype=np.float32, requires_grad=True)

        # PyTorch forward + backward
        z_torch = x_torch.expand(-1, 3, -1)
        z_torch.sum().backward()
        
        # Check gradients
        assert x_torch.grad is not None, "Gradient for x is None"

        # Detach gradients to numpy
        grad_x_torch = x_torch.grad.detach().numpy()

        # Custom forward + backward
        z_custom = x_custom.expand(-1, 3, -1)
        z_custom.sum().backward()
        
        # Check gradients
        assert x_custom.grad is not None, "Gradient for x is None"
        
        # Detach gradients to numpy
        grad_x_custom = x_custom.grad

        # Assert gradients are equal
        self.assertTrue(
            np.allclose(grad_x_custom, grad_x_torch, atol=1e-6),
            msg=(
                f"❌ Backward gradients differ!\n"
                f"Custom grad shape: {grad_x_custom.shape}, Torch grad shape: {grad_x_torch.shape}\n"
                f"Custom grad: {grad_x_custom}\n"
                f"Torch grad: {grad_x_torch}"
            )
        )


    def test_expand_backward_add_dimension(self):
        """
        Test that expand backward works correctly when adding new dimensions.
        """
        
        # Create random input data
        x_raw = np.random.rand(3, 1).astype(np.float32)
        
        # Setup tensors with grad
        x_torch = torch.tensor(x_raw, dtype=torch.float32, requires_grad=True)
        
        # Custom tensor
        x_custom = Tensor(x_raw, dtype=np.float32, requires_grad=True)

        # PyTorch forward + backward
        z_torch = x_torch.expand(2, 3, 4)
        z_torch.sum().backward()
        
        # Check gradients
        assert x_torch.grad is not None, "Gradient for x is None"

        # Detach gradients to numpy
        grad_x_torch = x_torch.grad.detach().numpy()

        # Custom forward + backward
        z_custom = x_custom.expand(2, 3, 4)
        z_custom.sum().backward()
        
        # Check gradients
        assert x_custom.grad is not None, "Gradient for x is None"
        
        # Detach gradients to numpy
        grad_x_custom = x_custom.grad

        # Assert gradients are equal
        self.assertTrue(
            np.allclose(grad_x_custom, grad_x_torch, atol=1e-6),
            msg=(
                f"❌ Backward gradients differ!\n"
                f"Custom grad shape: {grad_x_custom.shape}, Torch grad shape: {grad_x_torch.shape}\n"
                f"Custom grad: {grad_x_custom}\n"
                f"Torch grad: {grad_x_torch}"
            )
        )


    def test_expand_backward_complex(self):
        """
        Test expand backward with a complex computation graph.
        """
        
        # Create random input data
        x_raw = np.random.rand(1, 1, 4).astype(np.float32)
        
        # Setup tensors with grad
        x_torch = torch.tensor(x_raw, dtype=torch.float32, requires_grad=True)
        
        # Custom tensor
        x_custom = Tensor(x_raw, dtype=np.float32, requires_grad=True)

        # PyTorch forward + backward with computation
        z_torch = x_torch.expand(2, 3, 4)
        result_torch = (z_torch * 2 + 1).sum()
        result_torch.backward()
        
        # Check gradients
        assert x_torch.grad is not None, "Gradient for x is None"

        # Detach gradients to numpy
        grad_x_torch = x_torch.grad.detach().numpy()

        # Custom forward + backward with computation
        z_custom = x_custom.expand(2, 3, 4)
        result_custom = (z_custom * 2 + 1).sum()
        result_custom.backward()
        
        # Check gradients
        assert x_custom.grad is not None, "Gradient for x is None"
        
        # Detach gradients to numpy
        grad_x_custom = x_custom.grad

        # Assert gradients are equal
        self.assertTrue(
            np.allclose(grad_x_custom, grad_x_torch, atol=1e-6),
            msg=(
                f"❌ Backward gradients differ!\n"
                f"Custom grad shape: {grad_x_custom.shape}, Torch grad shape: {grad_x_torch.shape}\n"
                f"Custom grad: {grad_x_custom}\n"
                f"Torch grad: {grad_x_torch}"
            )
        )


    def test_expand_forward_4d_tensor(self):
        """
        Test expand on 4D tensors (common in attention mechanisms).
        Expand (B, S, 1, D) to (B, S, H, D).
        """
        
        # Create random input data: (batch=2, seq=4, 1, dim=8)
        x_raw = np.random.rand(2, 4, 1, 8).astype(np.float32)
        
        # Custom tensor
        x_tensor = Tensor(x_raw, dtype=np.float32, requires_grad=False)

        # Torch tensor
        x_torch = torch.tensor(x_raw, dtype=torch.float32, requires_grad=False)
        
        # Expand to (2, 4, 3, 8) - adding 3 heads
        out_torch = x_torch.expand(-1, -1, 3, -1).numpy()
        out_custom = x_tensor.expand(-1, -1, 3, -1).detach().to_numpy()

        # Assert values are close
        self.assertTrue(
            np.allclose(out_custom, out_torch, atol=1e-6),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom shape: {out_custom.shape}, Torch shape: {out_torch.shape}"
            )
        )


    def test_expand_backward_4d_tensor(self):
        """
        Test expand backward on 4D tensors.
        """
        
        # Create random input data: (batch=2, seq=4, 1, dim=8)
        x_raw = np.random.rand(2, 4, 1, 8).astype(np.float32)
        
        # Setup tensors with grad
        x_torch = torch.tensor(x_raw, dtype=torch.float32, requires_grad=True)
        x_custom = Tensor(x_raw, dtype=np.float32, requires_grad=True)

        # PyTorch forward + backward
        z_torch = x_torch.expand(-1, -1, 3, -1)
        z_torch.sum().backward()
        
        # Ensure the torch gradient is defined
        assert x_torch.grad is not None, "Gradient for x is None"
        
        grad_x_torch = x_torch.grad.detach().numpy()

        # Custom forward + backward
        z_custom = x_custom.expand(-1, -1, 3, -1)
        z_custom.sum().backward()
        grad_x_custom = x_custom.grad
        
        # Ensure the custom gradient is defined
        assert grad_x_custom is not None, "Gradient for x is None"

        # Assert gradients are equal
        self.assertTrue(
            np.allclose(grad_x_custom, grad_x_torch, atol=1e-6),
            msg=(
                f"❌ Backward gradients differ!\n"
                f"Custom grad shape: {grad_x_custom.shape}, Torch grad shape: {grad_x_torch.shape}\n"
                f"Custom grad: {grad_x_custom}\n"
                f"Torch grad: {grad_x_torch}"
            )
        )


    def test_expand_performance(self):
        """
        Test that expand performs reasonably well on larger tensors.
        """
        
        # Set the number of iterations for performance testing
        n_iters = 100
        
        # Create larger input data
        x_raw = np.random.rand(1, 512, 1, 64).astype(np.float32)
        
        # Custom tensor
        x_tensor = Tensor(x_raw, dtype=np.float32, requires_grad=True)

        # Torch tensor
        x_torch = torch.tensor(x_raw, dtype=torch.float32, requires_grad=True)
        
        # Time PyTorch
        start_torch = time.time()
        for _ in range(n_iters):
            z_torch = x_torch.expand(-1, -1, 8, -1)
            z_torch.sum().backward()
            x_torch.grad = None
        time_torch = time.time() - start_torch

        # Time custom implementation
        start_custom = time.time()
        for _ in range(n_iters):
            z_custom = x_tensor.expand(-1, -1, 8, -1)
            z_custom.sum().backward()
            x_tensor.grad = None
        time_custom = time.time() - start_custom
        
        # Ratio of custom to torch
        ratio_bwd = time_custom / time_torch if time_torch > 0 else float('inf')
        
        # Assert performance is within factor
        self.assertLess(
            ratio_bwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Kernel too slow: {ratio_bwd:.2f}x slower --> "
                f"torch.expand backward: {time_torch:.6f}s, custom: {time_custom:.6f}s"
            )
        )
        
        
if __name__ == "__main__":
    unittest.main()