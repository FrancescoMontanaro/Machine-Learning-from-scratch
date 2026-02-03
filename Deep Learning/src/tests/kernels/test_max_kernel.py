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


class TestMaxKernel(Test):

    def setUp(self):
        """
        Set up the test case.
        """

        # Create random input data
        base = np.random.randn(1000, 100).astype(np.float32)
        offsets = (np.arange(100).astype(np.float32) * 1e-3)
        self.x_raw = base + offsets


    def test_max_forward_values(self):
        """
        Test that max_forward matches torch.max on forward pass.
        """

        # Custom tensors
        x_custom = Tensor(self.x_raw, dtype=np.float32, requires_grad=False)

        # Torch tensors
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=False)

        # PyTorch forward
        out_torch = torch.max(x_torch, dim=1).values.detach().numpy()

        # Custom forward
        out_custom = x_custom.max(axis=1).detach().to_numpy()

        # Assert values are close
        self.assertTrue(
            np.allclose(out_custom, out_torch, atol=1e-6),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom: {out_custom}\n"
                f"Torch: {out_torch}"
            )
        )


    def test_max_backward_values(self):
        """
        Test that max_backward matches torch.max backward gradients.
        """

        # Setup tensors with grad
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)

        # Custom tensors
        x_custom = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)

        # PyTorch forward + backward
        out_torch = torch.max(x_torch, dim=1).values
        out_torch.backward(torch.ones_like(out_torch))

        # Custom forward + backward
        out_custom = x_custom.max(axis=1)
        out_custom.backward()

        # Check gradients
        assert x_torch.grad is not None, "Gradient for torch input is None"
        assert x_custom.grad is not None, "Gradient for custom input is None"

        # Detach gradients to numpy
        grad_x_torch = x_torch.grad.detach().numpy()

        # Detach gradients to numpy
        grad_x_custom = x_custom.grad

        # Assert gradients are equal
        self.assertTrue(
            np.allclose(grad_x_custom, grad_x_torch, atol=1e-6),
            msg=(
                f"❌ Backward gradients differ!\n"
                f"Custom:\n{grad_x_custom}\n"
                f"Torch:\n{grad_x_torch}"
            )
        )


    def test_max_performance(self):
        """
        Test to compare the performance
        """

        # Number of iterations for performance test
        n_iters = 10000

        # PyTorch forward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32)
        for _ in range(n_iters):
            _ = torch.max(x, dim=1).values
        t_torch_fwd = time.time() - start

        # Custom forward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32)
        for _ in range(n_iters):
            _ = x.max(axis=1)
        t_custom_fwd = time.time() - start

        # Print performance results
        print(f"torch.max forward: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s")

        # Assert forward speed is within factor
        ratio_fwd = t_custom_fwd / t_torch_fwd if t_torch_fwd > 0 else float('inf')
        self.assertLess(
            ratio_fwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Forward kernel too slow: {ratio_fwd:.2f}x slower --> "
                f"torch.max: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s"
            )
        )

        # PyTorch backward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)
        for _ in range(n_iters):
            out = torch.max(x, dim=1).values
            out.backward(torch.ones_like(out))
        t_torch_bwd = time.time() - start

        # Custom backward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)
        for _ in range(n_iters):
            out = x.max(axis=1)
            out.backward()
        t_custom_bwd = time.time() - start

        # Print performance results
        print(f"torch.max backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s")

        # Assert backward speed is within factor
        ratio_bwd = t_custom_bwd / t_torch_bwd if t_torch_bwd > 0 else float('inf')
        self.assertLess(
            ratio_bwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Backward kernel too slow: {ratio_bwd:.2f}x slower --> "
                f"torch.max backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s"
            )
        )


if __name__ == '__main__':
    unittest.main()
