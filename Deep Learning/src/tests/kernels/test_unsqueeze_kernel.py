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


class TestUnsqueezeKernel(Test):

    def setUp(self):
        """
        Set up the test case.
        """

        # Create random input data
        self.x_raw = np.random.randn(50, 20).astype(np.float32)


    def test_unsqueeze_forward_values(self):
        """
        Test that unsqueeze_forward matches torch.unsqueeze on forward pass.
        """

        # Custom tensors
        x_custom = Tensor(self.x_raw, dtype=np.float32, requires_grad=False)

        # Torch tensors
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=False)

        # PyTorch forward
        out_torch = torch.unsqueeze(x_torch, dim=1).detach().numpy()

        # Custom forward
        out_custom = x_custom.unsqueeze(axis=1).detach().to_numpy()

        # Assert values are close
        self.assertTrue(
            np.allclose(out_custom, out_torch, atol=1e-6),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom: {out_custom}\n"
                f"Torch: {out_torch}"
            )
        )


    def test_unsqueeze_backward_values(self):
        """
        Test that unsqueeze_backward matches torch.unsqueeze backward gradients.
        """

        # Setup tensors with grad
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)

        # Custom tensors
        x_custom = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)

        # PyTorch forward + backward
        out_torch = torch.unsqueeze(x_torch, dim=1)
        out_torch.backward(torch.ones_like(out_torch))

        # Custom forward + backward
        out_custom = x_custom.unsqueeze(axis=1)
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


    def test_unsqueeze_performance(self):
        """
        Test to compare the performance
        """

        # Number of iterations for performance test
        n_iters = 10000

        # PyTorch forward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32)
        for _ in range(n_iters):
            _ = torch.unsqueeze(x, dim=1)
        t_torch_fwd = time.time() - start

        # Custom forward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32)
        for _ in range(n_iters):
            _ = x.unsqueeze(axis=1)
        t_custom_fwd = time.time() - start

        # Print performance results
        print(f"torch.unsqueeze forward: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s")

        # Assert forward speed is within factor
        ratio_fwd = t_custom_fwd / t_torch_fwd if t_torch_fwd > 0 else float('inf')
        self.assertLess(
            ratio_fwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Forward kernel too slow: {ratio_fwd:.2f}x slower --> "
                f"torch.unsqueeze: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s"
            )
        )

        # PyTorch backward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)
        for _ in range(n_iters):
            out = torch.unsqueeze(x, dim=1)
            out.backward(torch.ones_like(out))
        t_torch_bwd = time.time() - start

        # Custom backward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)
        for _ in range(n_iters):
            out = x.unsqueeze(axis=1)
            out.backward()
        t_custom_bwd = time.time() - start

        # Print performance results
        print(f"torch.unsqueeze backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s")

        # Assert backward speed is within factor
        ratio_bwd = t_custom_bwd / t_torch_bwd if t_torch_bwd > 0 else float('inf')
        self.assertLess(
            ratio_bwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Backward kernel too slow: {ratio_bwd:.2f}x slower --> "
                f"torch.unsqueeze backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s"
            )
        )


if __name__ == '__main__':
    unittest.main()
