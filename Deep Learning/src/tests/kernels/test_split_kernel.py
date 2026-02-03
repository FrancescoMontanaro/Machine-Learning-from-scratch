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


class TestSplitKernel(Test):

    def setUp(self):
        """
        Set up the test case.
        """

        # Create random input data
        self.x_raw = np.random.randn(200, 60).astype(np.float32)
        self.sections = 3
        self.axis = 1


    def test_split_forward_values(self):
        """
        Test that split_forward matches torch.split on forward pass.
        """

        # Custom tensors
        x_custom = Tensor(self.x_raw, dtype=np.float32, requires_grad=False)

        # Torch tensors
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=False)

        # PyTorch forward
        out_torch = torch.split(x_torch, self.x_raw.shape[self.axis] // self.sections, dim=self.axis)

        # Custom forward
        out_custom = Tensor.split(x_custom, self.sections, axis=self.axis)

        for idx in range(self.sections):
            # Assert values are close
            self.assertTrue(
                np.allclose(out_custom[idx].detach().to_numpy(), out_torch[idx].detach().numpy(), atol=1e-6),
                msg=(
                    f"❌ Forward outputs differ at index {idx}!\n"
                    f"Custom: {out_custom[idx].detach().to_numpy()}\n"
                    f"Torch: {out_torch[idx].detach().numpy()}"
                )
            )


    def test_split_backward_values(self):
        """
        Test that split_backward matches torch.split backward gradients.
        """

        # Setup tensors with grad
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)

        # Custom tensors
        x_custom = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)

        # PyTorch forward + backward
        out_torch = torch.split(x_torch, self.x_raw.shape[self.axis] // self.sections, dim=self.axis)
        sum_torch = sum(out_torch)
        sum_torch.backward(torch.ones_like(sum_torch))

        # Custom forward + backward
        out_custom = Tensor.split(x_custom, self.sections, axis=self.axis)
        sum_custom = out_custom[0]
        for t in out_custom[1:]:
            sum_custom = sum_custom + t
        sum_custom.backward()

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


    def test_split_performance(self):
        """
        Test to compare the performance
        """

        # Number of iterations for performance test
        n_iters = 2000

        # PyTorch forward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32)
        for _ in range(n_iters):
            _ = torch.split(x, self.x_raw.shape[self.axis] // self.sections, dim=self.axis)
        t_torch_fwd = time.time() - start

        # Custom forward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32)
        for _ in range(n_iters):
            _ = Tensor.split(x, self.sections, axis=self.axis)
        t_custom_fwd = time.time() - start

        # Print performance results
        print(f"torch.split forward: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s")

        # Assert forward speed is within factor
        ratio_fwd = t_custom_fwd / t_torch_fwd if t_torch_fwd > 0 else float('inf')
        self.assertLess(
            ratio_fwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Forward kernel too slow: {ratio_fwd:.2f}x slower --> "
                f"torch.split: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s"
            )
        )

        # PyTorch backward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)
        for _ in range(n_iters):
            outs = torch.split(x, self.x_raw.shape[self.axis] // self.sections, dim=self.axis)
            total = sum(outs)
            total.backward(torch.ones_like(total))
        t_torch_bwd = time.time() - start

        # Custom backward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)
        for _ in range(n_iters):
            outs = Tensor.split(x, self.sections, axis=self.axis)
            total = outs[0]
            for t in outs[1:]:
                total = total + t
            total.backward()
        t_custom_bwd = time.time() - start

        # Print performance results
        print(f"torch.split backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s")

        # Assert backward speed is within factor
        ratio_bwd = t_custom_bwd / t_torch_bwd if t_torch_bwd > 0 else float('inf')
        self.assertLess(
            ratio_bwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Backward kernel too slow: {ratio_bwd:.2f}x slower --> "
                f"torch.split backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s"
            )
        )


if __name__ == '__main__':
    unittest.main()
