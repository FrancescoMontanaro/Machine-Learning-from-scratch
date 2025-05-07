import os
import sys
import time
import torch
import unittest
import numpy as np

# Ensure the src directory is on sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.core.functional.kernel.sub import sub_forward, sub_backward_a, sub_backward_b


class TestSubKernel(unittest.TestCase):
    
    def setUp(self):
        """
        Set up the test case.
        """

        # Create random input data
        self.x_raw = np.random.rand(1000, 100).astype(np.float32)
        self.y_raw = np.random.rand(1000, 100).astype(np.float32)

        # Torch tensors
        self.x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=False)
        self.y_torch = torch.tensor(self.y_raw, dtype=torch.float32, requires_grad=False)


    def test_sub_forward_values(self):
        """
        Test that sub_forward matches torch.sub on forward pass.
        """
        
        # PyTorch forward
        z_torch = torch.sub(self.x_torch, self.y_torch)
        out_torch = z_torch.detach().numpy()

        # Custom forward
        out_custom = sub_forward(self.x_raw, self.y_raw)

        # Assert values are close
        self.assertTrue(
            np.allclose(out_custom, out_torch, atol=1e-6),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom: {out_custom}\n"
                f"Torch: {out_torch}"
            )
        )
        

    def test_sub_backward_values(self):
        """
        Test that sub_forward matches torch.sub backward gradients.
        """
        
        # Setup tensors with grad
        x = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)
        y = torch.tensor(self.y_raw, dtype=torch.float32, requires_grad=True)

        # PyTorch forward + backward
        z = torch.sub(x, y)
        grad_output = torch.ones_like(z)
        z.backward(grad_output)
        
        # Check gradients
        assert x.grad is not None, "Gradient for z is None"
        assert y.grad is not None, "Gradient for z is None"

        # Detach gradients to numpy
        grad_x_torch = x.grad.detach().numpy()
        grad_y_torch = y.grad.detach().numpy()

        # Custom backward
        out_grad = np.ones_like(self.x_raw)
        grad_x_custom = np.zeros_like(self.x_raw)
        grad_y_custom = np.zeros_like(self.y_raw)

        # Custom backward pass
        sub_backward_a(out_grad, grad_x_custom, self.x_raw.shape)
        sub_backward_b(out_grad, grad_y_custom, self.y_raw.shape)

        # Assert gradients are equal
        self.assertTrue(
            np.allclose(grad_x_custom, grad_x_torch, atol=1e-6),
            msg = (
                f"❌ Backward grad_x differ!\n"
                f"Custom:\n{grad_x_custom}\n"
                f"Torch:\n{grad_x_torch}"
            )
        )
        self.assertTrue(
            np.allclose(grad_y_custom, grad_y_torch, atol=1e-6),
            msg = (
                f"❌ Backward grad_y differ!\n"
                f"Custom:\n{grad_y_custom}\n"
                f"Torch:\n{grad_y_torch}"
            )
        )


    def test_performance(self):
        """
        Test to compare the performance
        """
        
        # Number of iterations for performance test
        n_iters = 10000

        # PyTorch forward
        start = time.time()
        for _ in range(n_iters):
            x = torch.tensor(self.x_raw, dtype=torch.float32)
            y = torch.tensor(self.y_raw, dtype=torch.float32)
            _ = torch.sub(x, y)
        t_torch_fwd = time.time() - start

        # Custom forward
        start = time.time()
        for _ in range(n_iters):
            x = np.array(self.x_raw, dtype=np.float32)
            y = np.array(self.y_raw, dtype=np.float32)
            _ = sub_forward(x, y)
        t_custom_fwd = time.time() - start

        # Print performance results
        print(f"torch.sub forward: {t_torch_fwd:.6f}s, sub_forward: {t_custom_fwd:.6f}s")

        # Assert forward speed is within factor 3
        ratio_fwd = t_custom_fwd / t_torch_fwd if t_torch_fwd > 0 else float('inf')
        
        self.assertLess(
            ratio_fwd, 3,
            msg=(
                f"🟡 Forward kernel too slow: {ratio_fwd:.2f}x slower -->"
                f"torch.sub: {t_torch_fwd:.6f}s "
                f"sub_forward: {t_custom_fwd:.6f}s "
            )
        )

        # PyTorch backward
        start = time.time()
        for _ in range(n_iters):
            x = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)
            y = torch.tensor(self.y_raw, dtype=torch.float32, requires_grad=True)
            z = torch.sub(x, y)
            z.backward(torch.ones_like(z))
        t_torch_bwd = time.time() - start

        # Custom backward
        start = time.time()
        for _ in range(n_iters):
            x = np.array(self.x_raw, dtype=np.float32)
            out = sub_forward(x, self.y_raw)
            grad_buf = np.zeros_like(x)
            sub_backward_a(out, grad_buf, x.shape)
        t_custom_bwd = time.time() - start

        # Print performance results
        print(f"torch.sub backward: {t_torch_bwd:.6f}s, sub_forward: {t_custom_bwd:.6f}s")

        # Assert backward speed is within factor 3
        ratio_bwd = t_custom_bwd / t_torch_bwd if t_torch_bwd > 0 else float('inf')
        self.assertLess(
            ratio_bwd, 3,
            msg = (
                f"🟡 Backward kernel too slow: {ratio_bwd:.2f}x slower --> "
                f"torch.sub backward: {t_torch_bwd:.6f}s "
                f"sub_forward: {t_custom_bwd:.6f}s"
            )
        )


if __name__ == '__main__':
    unittest.main()