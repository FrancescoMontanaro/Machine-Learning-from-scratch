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


class TestConvTranspose2DKernel(Test):
    
    def setUp(self):
        """
        Set up the test case with random input and kernel.
        """
        
        # Create random input data
        # PyTorch format: (batch, channels, height, width)
        # Our format: (batch, height, width, channels)
        self.batch_size = 4
        self.in_channels = 3
        self.out_channels = 6
        self.in_height = 8
        self.in_width = 8
        self.kernel_size = (3, 3)
        self.stride = (2, 2)
        self.padding = (1, 1)
        self.output_padding = (1, 1)
        
        # Input data in PyTorch format
        self.x_raw = np.random.rand(
            self.batch_size, self.in_channels, self.in_height, self.in_width
        ).astype(np.float32)
        
        # Kernel in our format: (in_channels, out_channels, kH, kW)
        self.w_raw = np.random.rand(
            self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1]
        ).astype(np.float32)


    def test_conv_transpose_2d_forward_values(self):
        """
        Test that custom conv_transpose_2d forward matches torch.nn.functional.conv_transpose2d.
        """
        
        # Torch tensors
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=False)
        
        # PyTorch expects kernel shape: (in_channels, out_channels, kH, kW)
        w_torch = torch.tensor(self.w_raw, dtype=torch.float32, requires_grad=False)

        # Custom tensors - convert input to (batch, H, W, C) format
        x_custom = Tensor(self.x_raw.transpose(0, 2, 3, 1), dtype=np.float32, requires_grad=False)
        w_custom = Tensor(self.w_raw, dtype=np.float32, requires_grad=False)
        
        # PyTorch forward
        out_torch = F.conv_transpose2d(
            x_torch, w_torch, bias=None, 
            stride=self.stride, padding=self.padding, output_padding=self.output_padding
        )
        out_torch = out_torch.detach().numpy()

        # Custom forward
        out_custom = x_custom.conv_transpose_2d(
            w_custom, 
            stride=self.stride, 
            padding=self.padding,
            output_padding=self.output_padding
        )
        # Convert from (batch, H, W, C) to (batch, C, H, W) for comparison
        out_custom = out_custom.detach().to_numpy().transpose(0, 3, 1, 2)

        # Assert values are close
        self.assertTrue(
            np.allclose(out_custom, out_torch, atol=1e-5),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom shape: {out_custom.shape}\n"
                f"Torch shape:  {out_torch.shape}\n"
                f"Max diff: {np.max(np.abs(out_custom - out_torch))}"
            )
        )


    def test_conv_transpose_2d_backward_values(self):
        """
        Test that custom conv_transpose_2d gradients match torch autograd.
        """
        
        # Setup tensors with grad
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)
        w_torch = torch.tensor(self.w_raw, dtype=torch.float32, requires_grad=True)
        
        # Custom tensors
        x_custom = Tensor(self.x_raw.transpose(0, 2, 3, 1), dtype=np.float32, requires_grad=True)
        w_custom = Tensor(self.w_raw, dtype=np.float32, requires_grad=True)

        # PyTorch forward + backward
        z_torch = F.conv_transpose2d(
            x_torch, w_torch, bias=None,
            stride=self.stride, padding=self.padding, output_padding=self.output_padding
        )
        grad_output = torch.ones_like(z_torch)
        z_torch.backward(grad_output)
        
        # Check gradients
        assert x_torch.grad is not None, "Gradient for x is None"
        assert w_torch.grad is not None, "Gradient for w is None"

        # Detach gradients to numpy
        grad_x_torch = x_torch.grad.detach().numpy()
        grad_w_torch = w_torch.grad.detach().numpy()

        # Custom forward + backward
        z_custom = x_custom.conv_transpose_2d(
            w_custom, 
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding
        )
        z_custom.backward()
        
        # Check gradients
        assert x_custom.grad is not None, "Gradient for x is None"
        assert w_custom.grad is not None, "Gradient for w is None"
        
        # Detach gradients to numpy
        grad_w_custom = w_custom.grad
        # Convert x gradient from (batch, H, W, C) to (batch, C, H, W)
        grad_x_custom = x_custom.grad.transpose(0, 3, 1, 2)
        
        # Assert gradients are close
        self.assertTrue(
            np.allclose(grad_x_custom, grad_x_torch, atol=1e-5),
            msg = (
                f"❌ Backward grad_x differ!\n"
                f"Max diff: {np.max(np.abs(grad_x_custom - grad_x_torch))}"
            )
        )
        self.assertTrue(
            np.allclose(grad_w_custom, grad_w_torch, atol=1e-5),
            msg = (
                f"❌ Backward grad_w differ!\n"
                f"Max diff: {np.max(np.abs(grad_w_custom - grad_w_torch))}"
            )
        )


    def test_conv_transpose_2d_simple_stride1(self):
        """
        Test conv_transpose_2d with stride=1 and no padding (simplest case).
        """
        
        # Simple case
        x_raw = np.random.rand(2, 3, 4, 4).astype(np.float32)
        w_raw = np.random.rand(3, 2, 3, 3).astype(np.float32)
        
        # Torch
        x_torch = torch.tensor(x_raw, dtype=torch.float32)
        w_torch = torch.tensor(w_raw, dtype=torch.float32)
        out_torch = F.conv_transpose2d(x_torch, w_torch, stride=1, padding=0)
        out_torch = out_torch.detach().numpy()
        
        # Custom
        x_custom = Tensor(x_raw.transpose(0, 2, 3, 1), dtype=np.float32)
        w_custom = Tensor(w_raw, dtype=np.float32)
        out_custom = x_custom.conv_transpose_2d(w_custom, stride=(1, 1), padding=(0, 0))
        out_custom = out_custom.detach().to_numpy().transpose(0, 3, 1, 2)
        
        # Compare
        self.assertTrue(
            np.allclose(out_custom, out_torch, atol=1e-5),
            msg=f"Simple stride=1 test failed. Max diff: {np.max(np.abs(out_custom - out_torch))}"
        )


    def test_conv_transpose_2d_performance(self):
        """
        Compare performance of forward and backward.
        """
        
        # Set the number of iterations for performance testing
        n_iters = 50

        # PyTorch forward
        x = torch.tensor(self.x_raw, dtype=torch.float32)
        w = torch.tensor(self.w_raw, dtype=torch.float32)
        start = time.time()
        for _ in range(n_iters):
            _ = F.conv_transpose2d(x, w, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
        t_torch_fwd = time.time() - start

        # Custom forward
        x = Tensor(self.x_raw.transpose(0, 2, 3, 1), dtype=np.float32)
        w = Tensor(self.w_raw, dtype=np.float32)
        start = time.time()
        for _ in range(n_iters):
            _ = x.conv_transpose_2d(w, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
        t_custom_fwd = time.time() - start

        # Print performance
        print(f"torch.conv_transpose2d forward: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s")
        
        # Ratio of custom to torch
        ratio_fwd = t_custom_fwd / t_torch_fwd if t_torch_fwd > 0 else float('inf')
        
        # Assert performance is within factor
        self.assertLess(
            ratio_fwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Forward kernel too slow: {ratio_fwd:.2f}x slower --> "
                f"torch.conv_transpose2d: {t_torch_fwd:.6f}s, custom: {t_custom_fwd:.6f}s"
            )
        )

        # PyTorch backward
        x = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)
        w = torch.tensor(self.w_raw, dtype=torch.float32, requires_grad=True)
        start = time.time()
        for _ in range(n_iters):
            out = F.conv_transpose2d(x, w, bias=None, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
            out.backward(torch.ones_like(out))
        t_torch_bwd = time.time() - start

        # Custom backward
        x = Tensor(self.x_raw.transpose(0, 2, 3, 1), dtype=np.float32, requires_grad=True)
        w = Tensor(self.w_raw, dtype=np.float32, requires_grad=True)
        start = time.time()
        for _ in range(n_iters):
            out = x.conv_transpose_2d(w, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
            out.backward()
        t_custom_bwd = time.time() - start

        # Print performance
        print(f"torch.conv_transpose2d backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s")
        
        # Ratio of custom to torch
        ratio_bwd = t_custom_bwd / t_torch_bwd if t_torch_bwd > 0 else float('inf')
        
        # Assert performance is within factor
        self.assertLess(
            ratio_bwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Backward kernel too slow: {ratio_bwd:.2f}x slower --> "
                f"torch.conv_transpose2d backward: {t_torch_bwd:.6f}s, custom: {t_custom_bwd:.6f}s"
            )
        )


if __name__ == '__main__':
    unittest.main()
