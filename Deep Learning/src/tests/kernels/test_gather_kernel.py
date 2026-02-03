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


class TestGatherKernel(Test):

    def setUp(self):
        """
        Set up the test case.
        """

        # Create random input data
        self.x_raw = np.random.randn(200, 50).astype(np.float32)
        self.index_raw = np.random.randint(0, 50, size=(200, 20)).astype(np.int64)
        self.axis = 1


    def test_gather_forward_values(self):
        """
        Test that gather_forward matches torch.gather on forward pass.
        """

        # Custom tensors
        x_custom = Tensor(self.x_raw, dtype=np.float32, requires_grad=False)
        idx_custom = Tensor(self.index_raw, dtype=np.int64, requires_grad=False)

        # Torch tensors
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=False)
        idx_torch = torch.tensor(self.index_raw, dtype=torch.int64, requires_grad=False)

        # PyTorch forward
        out_torch = torch.gather(x_torch, self.axis, idx_torch).detach().numpy()

        # Custom forward
        out_custom = x_custom.gather(self.axis, idx_custom).detach().to_numpy()

        # Assert values are close
        self.assertTrue(
            np.allclose(out_custom, out_torch, atol=1e-6),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom: {out_custom}\n"
                f"Torch: {out_torch}"
            )
        )


    def test_gather_backward_values(self):
        """
        Test that gather_backward matches torch.gather backward gradients.
        """

        # Setup tensors with grad
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)
        idx_torch = torch.tensor(self.index_raw, dtype=torch.int64, requires_grad=False)

        # Custom tensors
        x_custom = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)
        idx_custom = Tensor(self.index_raw, dtype=np.int64, requires_grad=False)

        # PyTorch forward + backward
        out_torch = torch.gather(x_torch, self.axis, idx_torch)
        out_torch.backward(torch.ones_like(out_torch))

        # Custom forward + backward
        out_custom = x_custom.gather(self.axis, idx_custom)
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


    def test_gather_performance(self):
        """
        Test to compare the performance
        """

        # Number of iterations for performance test
        n_iters = 2000

        # PyTorch forward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32)
        idx = torch.tensor(self.index_raw, dtype=torch.int64)
        for _ in range(n_iters):
            _ = torch.gather(x, self.axis, idx)
        t_torch_fwd = time.time() - start

        # Custom forward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32)
        idx = Tensor(self.index_raw, dtype=np.int64)
        for _ in range(n_iters):
            _ = x.gather(self.axis, idx)
        t_custom_fwd = time.time() - start

        # Print performance results
        print(f"torch.gather forward: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s")

        # Assert forward speed is within factor
        ratio_fwd = t_custom_fwd / t_torch_fwd if t_torch_fwd > 0 else float('inf')
        self.assertLess(
            ratio_fwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Forward kernel too slow: {ratio_fwd:.2f}x slower --> "
                f"torch.gather: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s"
            )
        )

        # PyTorch backward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)
        idx = torch.tensor(self.index_raw, dtype=torch.int64)
        for _ in range(n_iters):
            out = torch.gather(x, self.axis, idx)
            out.backward(torch.ones_like(out))
        t_torch_bwd = time.time() - start

        # Custom backward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)
        idx = Tensor(self.index_raw, dtype=np.int64)
        for _ in range(n_iters):
            out = x.gather(self.axis, idx)
            out.backward()
        t_custom_bwd = time.time() - start

        # Print performance results
        print(f"torch.gather backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s")

        # Assert backward speed is within factor
        ratio_bwd = t_custom_bwd / t_torch_bwd if t_torch_bwd > 0 else float('inf')
        self.assertLess(
            ratio_bwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Backward kernel too slow: {ratio_bwd:.2f}x slower --> "
                f"torch.gather backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s"
            )
        )


if __name__ == '__main__':
    unittest.main()
