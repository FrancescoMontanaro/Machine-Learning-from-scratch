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


class TestEinsumKernel(Test):

    def setUp(self):
        """
        Set up the test case.
        """

        # Create random input data
        self.a_raw = np.random.randn(200, 30).astype(np.float32)
        self.b_raw = np.random.randn(30, 40).astype(np.float32)
        self.subscripts = 'ij,jk->ik'


    def test_einsum_forward_values(self):
        """
        Test that einsum_forward matches torch.einsum on forward pass.
        """

        # Custom tensors
        a_custom = Tensor(self.a_raw, dtype=np.float32, requires_grad=False)
        b_custom = Tensor(self.b_raw, dtype=np.float32, requires_grad=False)

        # Torch tensors
        a_torch = torch.tensor(self.a_raw, dtype=torch.float32, requires_grad=False)
        b_torch = torch.tensor(self.b_raw, dtype=torch.float32, requires_grad=False)

        # PyTorch forward
        out_torch = torch.einsum(self.subscripts, a_torch, b_torch).detach().numpy()

        # Custom forward
        out_custom = Tensor.einsum(self.subscripts, a_custom, b_custom).detach().to_numpy()

        # Assert values are close
        self.assertTrue(
            np.allclose(out_custom, out_torch, atol=1e-6),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom: {out_custom}\n"
                f"Torch: {out_torch}"
            )
        )


    def test_einsum_backward_values(self):
        """
        Test that einsum_backward matches torch.einsum backward gradients.
        """

        # Setup tensors with grad
        a_torch = torch.tensor(self.a_raw, dtype=torch.float32, requires_grad=True)
        b_torch = torch.tensor(self.b_raw, dtype=torch.float32, requires_grad=True)

        # Custom tensors
        a_custom = Tensor(self.a_raw, dtype=np.float32, requires_grad=True)
        b_custom = Tensor(self.b_raw, dtype=np.float32, requires_grad=True)

        # PyTorch forward + backward
        out_torch = torch.einsum(self.subscripts, a_torch, b_torch)
        out_torch.backward(torch.ones_like(out_torch))

        # Custom forward + backward
        out_custom = Tensor.einsum(self.subscripts, a_custom, b_custom)
        out_custom.backward()

        # Check gradients
        assert a_torch.grad is not None, "Gradient for torch input is None"
        assert b_torch.grad is not None, "Gradient for torch input is None"
        assert a_custom.grad is not None, "Gradient for custom input is None"
        assert b_custom.grad is not None, "Gradient for custom input is None"

        # Detach gradients to numpy
        grad_a_torch = a_torch.grad.detach().numpy()
        grad_b_torch = b_torch.grad.detach().numpy()

        # Detach gradients to numpy
        grad_a_custom = a_custom.grad
        grad_b_custom = b_custom.grad

        # Assert gradients are equal
        self.assertTrue(
            np.allclose(grad_a_custom, grad_a_torch, atol=1e-6),
            msg=(
                f"❌ Backward gradients differ for A!\n"
                f"Custom:\n{grad_a_custom}\n"
                f"Torch:\n{grad_a_torch}"
            )
        )

        self.assertTrue(
            np.allclose(grad_b_custom, grad_b_torch, atol=1e-6),
            msg=(
                f"❌ Backward gradients differ for B!\n"
                f"Custom:\n{grad_b_custom}\n"
                f"Torch:\n{grad_b_torch}"
            )
        )


    def test_einsum_performance(self):
        """
        Test to compare the performance
        """

        # Number of iterations for performance test
        n_iters = 1000

        # PyTorch forward
        start = time.time()
        a = torch.tensor(self.a_raw, dtype=torch.float32)
        b = torch.tensor(self.b_raw, dtype=torch.float32)
        for _ in range(n_iters):
            _ = torch.einsum(self.subscripts, a, b)
        t_torch_fwd = time.time() - start

        # Custom forward
        start = time.time()
        a = Tensor(self.a_raw, dtype=np.float32)
        b = Tensor(self.b_raw, dtype=np.float32)
        for _ in range(n_iters):
            _ = Tensor.einsum(self.subscripts, a, b)
        t_custom_fwd = time.time() - start

        # Print performance results
        print(f"torch.einsum forward: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s")

        # Assert forward speed is within factor
        ratio_fwd = t_custom_fwd / t_torch_fwd if t_torch_fwd > 0 else float('inf')
        self.assertLess(
            ratio_fwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Forward kernel too slow: {ratio_fwd:.2f}x slower --> "
                f"torch.einsum: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s"
            )
        )

        # PyTorch backward
        start = time.time()
        a = torch.tensor(self.a_raw, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(self.b_raw, dtype=torch.float32, requires_grad=True)
        for _ in range(n_iters):
            out = torch.einsum(self.subscripts, a, b)
            out.backward(torch.ones_like(out))
        t_torch_bwd = time.time() - start

        # Custom backward
        start = time.time()
        a = Tensor(self.a_raw, dtype=np.float32, requires_grad=True)
        b = Tensor(self.b_raw, dtype=np.float32, requires_grad=True)
        for _ in range(n_iters):
            out = Tensor.einsum(self.subscripts, a, b)
            out.backward()
        t_custom_bwd = time.time() - start

        # Print performance results
        print(f"torch.einsum backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s")

        # Assert backward speed is within factor
        ratio_bwd = t_custom_bwd / t_torch_bwd if t_torch_bwd > 0 else float('inf')
        self.assertLess(
            ratio_bwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Backward kernel too slow: {ratio_bwd:.2f}x slower --> "
                f"torch.einsum backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s"
            )
        )


if __name__ == '__main__':
    unittest.main()
