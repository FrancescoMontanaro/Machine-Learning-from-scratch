import os
import sys
import time
import torch
import unittest
import numpy as np
import torch.nn.functional as F

# Ensure the src directory is on sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src import Tensor
from src.tests.base import Test


class TestMaxPool2DKernel(Test):

    def setUp(self):
        """
        Set up the test case.
        """

        # Create random input data
        self.x_raw = np.random.rand(8, 16, 16, 3).astype(np.float32)
        self.kernel_size = (2, 2)
        self.stride = (2, 2)


    def test_max_pool_2d_forward_values(self):
        """
        Test that custom max_pool_2d forward matches torch.nn.functional.max_pool2d.
        """

        # Torch tensors
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=False).permute(0, 3, 1, 2)

        # Custom tensors
        x_custom = Tensor(self.x_raw, dtype=np.float32, requires_grad=False)

        # PyTorch forward
        out_torch = F.max_pool2d(x_torch, kernel_size=self.kernel_size, stride=self.stride)
        out_torch = out_torch.permute(0, 2, 3, 1).detach().numpy()

        # Custom forward
        out_custom = x_custom.max_pool_2d(kernel_size=self.kernel_size, stride=self.stride)
        out_custom = out_custom.detach().to_numpy()

        # Assert values are close
        self.assertTrue(
            np.allclose(out_custom, out_torch, atol=1e-6),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom: {out_custom.shape}\n"
                f"Torch: {out_torch.shape}"
            )
        )


    def test_max_pool_2d_backward_values(self):
        """
        Test that custom max_pool_2d gradients match torch autograd.
        """

        # Setup tensors with grad
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True).permute(0, 3, 1, 2)

        # Custom tensors
        x_custom = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)

        # PyTorch forward + backward
        out_torch = F.max_pool2d(x_torch, kernel_size=self.kernel_size, stride=self.stride)
        out_torch.backward(torch.ones_like(out_torch))

        # Custom forward + backward
        out_custom = x_custom.max_pool_2d(kernel_size=self.kernel_size, stride=self.stride)
        out_custom.backward()

        # Check gradients
        assert x_torch.grad is not None, "Gradient for torch input is None"
        assert x_custom.grad is not None, "Gradient for custom input is None"
        grad_torch = x_torch.grad.permute(0, 2, 3, 1).detach().numpy()

        # Detach gradients to numpy
        grad_x_torch = x_torch.grad.detach().numpy()

        # Detach gradients to numpy
        grad_x_custom = x_custom.grad

        # Assert gradients are equal
        self.assertTrue(
            np.allclose(x_custom.grad, grad_torch, atol=1e-6),
            msg=(
                f"❌ Backward gradients differ!\n"
                f"Custom:\n{grad_x_custom}\n"
                f"Torch:\n{grad_torch}"
            )
        )


    def test_max_pool_2d_performance(self):
        """
        Test to compare the performance
        """

        # Number of iterations for performance test
        n_iters = 500

        # PyTorch forward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32).permute(0, 3, 1, 2)
        for _ in range(n_iters):
            _ = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        t_torch_fwd = time.time() - start

        # Custom forward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32)
        for _ in range(n_iters):
            _ = x.max_pool_2d(kernel_size=self.kernel_size, stride=self.stride)
        t_custom_fwd = time.time() - start

        # Print performance results
        print(f"torch.max_pool2d forward: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s")

        # Assert forward speed is within factor
        ratio_fwd = t_custom_fwd / t_torch_fwd if t_torch_fwd > 0 else float('inf')
        self.assertLess(
            ratio_fwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Forward kernel too slow: {ratio_fwd:.2f}x slower --> "
                f"torch.max_pool2d: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s"
            )
        )

        # PyTorch backward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True).permute(0, 3, 1, 2)
        for _ in range(n_iters):
            out = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
            out.backward(torch.ones_like(out))
        t_torch_bwd = time.time() - start

        # Custom backward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)
        for _ in range(n_iters):
            out = x.max_pool_2d(kernel_size=self.kernel_size, stride=self.stride)
            out.backward()
        t_custom_bwd = time.time() - start

        # Print performance results
        print(f"torch.max_pool2d backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s")

        # Assert backward speed is within factor
        ratio_bwd = t_custom_bwd / t_torch_bwd if t_torch_bwd > 0 else float('inf')
        self.assertLess(
            ratio_bwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Backward kernel too slow: {ratio_bwd:.2f}x slower --> "
                f"torch.max_pool2d backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s"
            )
        )


if __name__ == '__main__':
    unittest.main()
