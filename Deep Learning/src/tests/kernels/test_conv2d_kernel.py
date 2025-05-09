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


class TestConv2DKernel(Test):
    
    def setUp(self):
        """
        Set up the test case with random input and kernel.
        """
        
        # Create random input data
        self.x_raw = np.random.rand(8, 3, 32, 32).astype(np.float32)
        self.w_raw = np.random.rand(6, 3, 5, 5).astype(np.float32)
        self.stride = 1


    def test_conv2d_forward_values(self):
        """
        Test that custom conv2d forward matches torch.nn.functional.conv2d.
        """
        
        # Torch tensors
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=False)
        w_torch = torch.tensor(self.w_raw, dtype=torch.float32, requires_grad=False)

        # Custom tensors
        x_custom = Tensor(self.x_raw.transpose(0, 2, 3, 1), dtype=np.float32, requires_grad=False)
        w_custom = Tensor(self.w_raw, dtype=np.float32, requires_grad=False)
        
        # PyTorch forward
        out_torch = F.conv2d(x_torch, w_torch, bias=None, stride=self.stride)
        out_torch = out_torch.detach().numpy()

        # Custom forward
        out_custom = x_custom.conv_2d(w_custom, stride=(self.stride, self.stride))
        out_custom = out_custom.detach().to_numpy().transpose(0, 3, 1, 2)

        # Assert values are close
        self.assertTrue(
            np.allclose(out_custom, out_torch, atol=1e-6),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom: {out_custom.shape}\n"
                f"Torch:  {out_torch.shape}"
            )
        )


    def test_conv2d_backward_values(self):
        """
        Test that custom conv2d gradients match torch autograd.
        """
        
        # Setup tensors with grad
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)
        w_torch = torch.tensor(self.w_raw, dtype=torch.float32, requires_grad=True)
        
        # Custom tensors
        x_custom = Tensor(self.x_raw.transpose(0, 2, 3, 1), dtype=np.float32, requires_grad=True)
        w_custom = Tensor(self.w_raw, dtype=np.float32, requires_grad=True)

        # PyTorch forward + backward
        z_torch = F.conv2d(x_torch, w_torch, bias=None, stride=self.stride)
        grad_output = torch.ones_like(z_torch)
        z_torch.backward(grad_output)
        
        # Check gradients
        assert x_torch.grad is not None, "Gradient for z is None"
        assert w_torch.grad is not None, "Gradient for z is None"

        # Detach gradients to numpy
        grad_x_torch = x_torch.grad.detach().numpy()
        grad_w_torch = w_torch.grad.detach().numpy()

        # Custom forward + backward
        z_custom = x_custom.conv_2d(w_custom, stride=(self.stride, self.stride))
        z_custom.backward()
        
        # Check gradients
        assert x_custom.grad is not None, "Gradient for z is None"
        assert w_custom.grad is not None, "Gradient for z is None"
        
        # Detach gradients to numpy
        grad_w_custom = w_custom.grad
        grad_x_custom = x_custom.grad.transpose(0, 3, 1, 2)
        
        # Assert gradients are close
        self.assertTrue(
            np.allclose(grad_x_custom, grad_x_torch, atol=1e-6),
            msg = (
                f"❌ Backward grad_w differ!\n"
                f"Custom:\n{grad_x_custom}\n"
                f"Torch:\n{grad_x_torch}"
            )
        )
        self.assertTrue(
            np.allclose(grad_w_custom, grad_w_torch, atol=1e-6),
            msg = (
                f"❌ Backward grad_y differ!\n"
                f"Custom:\n{grad_w_custom}\n"
                f"Torch:\n{grad_w_torch}"
            )
        )


    def test_conv2d_performance(self):
        """
        Compare performance of forward and backward.
        """
        
        # Set the number of iterations for performance testing
        n_iters = 100

        # PyTorch forward
        x = torch.tensor(self.x_raw, dtype=torch.float32)
        w = torch.tensor(self.w_raw, dtype=torch.float32)
        start = time.time()
        for _ in range(n_iters):
            _ = F.conv2d(x, w, stride=self.stride)
        t_torch_fwd = time.time() - start

        # Custom forward
        x = Tensor(self.x_raw.transpose(0, 2, 3, 1), dtype=np.float32)
        w = Tensor(self.w_raw, dtype=np.float32)
        start = time.time()
        for _ in range(n_iters):
            _ = x.conv_2d(w, stride=(self.stride, self.stride))
        t_custom_fwd = time.time() - start

        # Print performance
        print(f"torch.conv2d forward: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s")
        
        # Ratio of custom to torch
        ratio_fwd = t_custom_fwd / t_torch_fwd if t_torch_fwd > 0 else float('inf')
        
        # Assert performance is within factor
        self.assertLess(
            ratio_fwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Forward kernel too slow: {ratio_fwd:.2f}x slower --> "
                f"torch.conv2d: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s"
            )
        )

        # PyTorch backward
        x = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)
        w = torch.tensor(self.w_raw, dtype=torch.float32, requires_grad=True)
        start = time.time()
        for _ in range(n_iters):
            out = F.conv2d(x, w, bias=None, stride=self.stride)
            out.backward(torch.ones_like(out))
        t_torch_bwd = time.time() - start

        # Custom backward
        x = Tensor(self.x_raw.transpose(0, 2, 3, 1), dtype=np.float32, requires_grad=True)
        w = Tensor(self.w_raw, dtype=np.float32, requires_grad=True)
        start = time.time()
        for _ in range(n_iters):
            out = x.conv_2d(w, stride=(self.stride, self.stride))
            out.backward()
        t_custom_bwd = time.time() - start

        # Print performance
        print(f"torch.conv2d backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s")
        
        # Ratio of custom to torch
        ratio_bwd = t_custom_bwd / t_torch_bwd if t_torch_bwd > 0 else float('inf')
        
        # Assert performance is within factor
        self.assertLess(
            ratio_bwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Backward kernel too slow: {ratio_bwd:.2f}x slower --> "
                f"torch.conv2d backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s"
            )
        )


if __name__ == '__main__':
    unittest.main()