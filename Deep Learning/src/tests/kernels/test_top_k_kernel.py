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


class TestTopKKernel(Test):

    def setUp(self):
        """
        Set up the test case.
        """

        # Create random input data
        base = np.random.rand(200, 50).astype(np.float32)
        offsets = (np.arange(50).astype(np.float32) * 1e-4)
        self.x_raw = base + offsets
        self.k = 5
        self.axis = 1


    def test_top_k_forward_values(self):
        """
        Test that top_k_forward matches torch.topk on forward pass.
        """

        # Custom tensors
        x_custom = Tensor(self.x_raw, dtype=np.float32, requires_grad=False)

        # Torch tensors
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=False)

        values_torch, indices_torch = torch.topk(x_torch, self.k, dim=self.axis, largest=True, sorted=True)
        values_custom, indices_custom = x_custom.top_k(self.k, axis=self.axis, largest=True, sorted=True)

        # Assert values are close
        self.assertTrue(
            np.allclose(values_custom.detach().to_numpy(), values_torch.detach().numpy(), atol=1e-6),
            msg=(
                f"❌ Forward values differ!\n"
                f"Custom: {values_custom.detach().to_numpy()}\n"
                f"Torch: {values_torch.detach().numpy()}"
            )
        )

        self.assertTrue(
            np.array_equal(indices_custom.detach().to_numpy(), indices_torch.detach().numpy()),
            msg=(
                f"❌ Forward indices differ!\n"
                f"Custom: {indices_custom.detach().to_numpy()}\n"
                f"Torch: {indices_torch.detach().numpy()}"
            )
        )


    def test_top_k_backward_values(self):
        """
        Test that top_k_backward matches torch.topk backward gradients.
        """

        # Setup tensors with grad
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)

        # Custom tensors
        x_custom = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)

        values_torch, _ = torch.topk(x_torch, self.k, dim=self.axis, largest=True, sorted=True)
        values_torch.backward(torch.ones_like(values_torch))

        values_custom, _ = x_custom.top_k(self.k, axis=self.axis, largest=True, sorted=True)
        values_custom.backward()

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


    def test_top_k_performance(self):
        """
        Test to compare the performance
        """

        # Number of iterations for performance test
        n_iters = 1000

        # PyTorch forward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32)
        for _ in range(n_iters):
            _ = torch.topk(x, self.k, dim=self.axis, largest=True, sorted=True)
        t_torch_fwd = time.time() - start

        # Custom forward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32)
        for _ in range(n_iters):
            _ = x.top_k(self.k, axis=self.axis, largest=True, sorted=True)
        t_custom_fwd = time.time() - start

        # Print performance results
        print(f"torch.topk forward: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s")

        # Assert forward speed is within factor
        ratio_fwd = t_custom_fwd / t_torch_fwd if t_torch_fwd > 0 else float('inf')
        self.assertLess(
            ratio_fwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Forward kernel too slow: {ratio_fwd:.2f}x slower --> "
                f"torch.topk: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s"
            )
        )

        # PyTorch backward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)
        for _ in range(n_iters):
            values, _ = torch.topk(x, self.k, dim=self.axis, largest=True, sorted=True)
            values.backward(torch.ones_like(values))
        t_torch_bwd = time.time() - start

        # Custom backward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)
        for _ in range(n_iters):
            values, _ = x.top_k(self.k, axis=self.axis, largest=True, sorted=True)
            values.backward()
        t_custom_bwd = time.time() - start

        # Print performance results
        print(f"torch.topk backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s")

        # Assert backward speed is within factor
        ratio_bwd = t_custom_bwd / t_torch_bwd if t_torch_bwd > 0 else float('inf')
        self.assertLess(
            ratio_bwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Backward kernel too slow: {ratio_bwd:.2f}x slower --> "
                f"torch.topk backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s"
            )
        )


if __name__ == '__main__':
    unittest.main()
