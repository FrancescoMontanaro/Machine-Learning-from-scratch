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


class TestPadKernel(Test):

    def setUp(self):
        """
        Set up the test case.
        """

        # Create random input data
        self.x_raw = np.random.randn(8, 16, 16, 3).astype(np.float32)
        self.pad_width = ((0, 0), (1, 2), (3, 4), (0, 0))


    def test_pad_forward_values(self):
        """
        Test that pad_forward matches torch.nn.functional.pad on forward pass.
        """

        # Custom tensors
        x_custom = Tensor(self.x_raw, dtype=np.float32, requires_grad=False)

        # Torch tensors
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=False).permute(0, 3, 1, 2)

        (_, _), (pt, pb), (pl, pr), (_, _) = self.pad_width

        # PyTorch forward
        out_torch = F.pad(x_torch, (pl, pr, pt, pb)).permute(0, 2, 3, 1).detach().numpy()

        # Custom forward
        out_custom = x_custom.pad(self.pad_width).detach().to_numpy()

        # Assert values are close
        self.assertTrue(
            np.allclose(out_custom, out_torch, atol=1e-6),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom: {out_custom.shape}\n"
                f"Torch: {out_torch.shape}"
            )
        )


    def test_pad_backward_values(self):
        """
        Test that pad_backward matches torch.nn.functional.pad backward gradients.
        """

        # Setup tensors with grad
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True).permute(0, 3, 1, 2)

        # Custom tensors
        x_custom = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)

        (_, _), (pt, pb), (pl, pr), (_, _) = self.pad_width

        # PyTorch forward + backward
        out_torch = F.pad(x_torch, (pl, pr, pt, pb))
        out_torch.backward(torch.ones_like(out_torch))

        # Custom forward + backward
        out_custom = x_custom.pad(self.pad_width)
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


    def test_pad_performance(self):
        """
        Test to compare the performance
        """

        # Number of iterations for performance test
        n_iters = 1000

        (_, _), (pt, pb), (pl, pr), (_, _) = self.pad_width

        # PyTorch forward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32).permute(0, 3, 1, 2)
        for _ in range(n_iters):
            _ = F.pad(x, (pl, pr, pt, pb))
        t_torch_fwd = time.time() - start

        # Custom forward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32)
        for _ in range(n_iters):
            _ = x.pad(self.pad_width)
        t_custom_fwd = time.time() - start

        # Print performance results
        print(f"torch.pad forward: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s")

        # Assert forward speed is within factor
        ratio_fwd = t_custom_fwd / t_torch_fwd if t_torch_fwd > 0 else float('inf')
        self.assertLess(
            ratio_fwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Forward kernel too slow: {ratio_fwd:.2f}x slower --> "
                f"torch.pad: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s"
            )
        )

        # PyTorch backward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True).permute(0, 3, 1, 2)
        for _ in range(n_iters):
            out = F.pad(x, (pl, pr, pt, pb))
            out.backward(torch.ones_like(out))
        t_torch_bwd = time.time() - start

        # Custom backward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)
        for _ in range(n_iters):
            out = x.pad(self.pad_width)
            out.backward()
        t_custom_bwd = time.time() - start

        # Print performance results
        print(f"torch.pad backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s")

        # Assert backward speed is within factor
        ratio_bwd = t_custom_bwd / t_torch_bwd if t_torch_bwd > 0 else float('inf')
        self.assertLess(
            ratio_bwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Backward kernel too slow: {ratio_bwd:.2f}x slower --> "
                f"torch.pad backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s"
            )
        )


if __name__ == '__main__':
    unittest.main()
