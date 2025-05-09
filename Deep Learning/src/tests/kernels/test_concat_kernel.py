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


class TestConcatKernel(Test):
    
    def setUp(self):
        """
        Set up the test case.
        """

        # Create random input data
        # Tensors share all dimensions except the one along which we concatenate
        self.x_raw = np.random.rand(100, 30, 40).astype(np.float32)
        self.y_raw = np.random.rand(100, 30, 40).astype(np.float32)

        # Concatenate along this axis
        self.concat_axis = 0


    def test_concat_forward_values(self):
        """
        Test that concat_forward matches torch.cat on forward pass.
        """
        
        # Custom tensors
        x_tensor = Tensor(self.x_raw, dtype=np.float32, requires_grad=False)
        y_tensor = Tensor(self.y_raw, dtype=np.float32, requires_grad=False)

        # Torch tensors
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=False)
        y_torch = torch.tensor(self.y_raw, dtype=torch.float32, requires_grad=False)
        
        # PyTorch forward
        z_torch = torch.cat([x_torch, y_torch], dim=self.concat_axis)
        out_torch = z_torch.detach().numpy()

        # Custom forward
        out_custom = Tensor.concat([x_tensor, y_tensor], axis=self.concat_axis)
        out_custom = out_custom.detach().to_numpy()

        # Assert values are close
        self.assertTrue(
            np.allclose(out_custom, out_torch, atol=1e-1),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom: {out_custom}\n"
                f"Torch: {out_torch}"
            )
        )
        
        
    def test_concat_backward_values(self):
        """
        Test that concat_forward matches torch.cat backward gradients.
        """
        
        # Setup tensors with grad
        x_torch = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)
        y_torch = torch.tensor(self.y_raw, dtype=torch.float32, requires_grad=True)
        
        # Custom tensors
        x_custom = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)
        y_custom = Tensor(self.y_raw, dtype=np.float32, requires_grad=True)

        # PyTorch forward + backward
        z_torch = torch.cat([x_torch, y_torch], dim=self.concat_axis)
        grad_output = torch.ones_like(z_torch)
        z_torch.backward(grad_output)
        
        # Check gradients
        assert x_torch.grad is not None, "Gradient for z is None"
        assert y_torch.grad is not None, "Gradient for z is None"

        # Detach gradients to numpy
        grad_x_torch = x_torch.grad.detach().numpy()
        grad_y_torch = y_torch.grad.detach().numpy()

        # Custom forward + backward
        z_custom = Tensor.concat([x_custom, y_custom], axis=self.concat_axis)
        z_custom.backward()
        
        # Check gradients
        assert x_custom.grad is not None, "Gradient for z is None"
        assert y_custom.grad is not None, "Gradient for z is None"
        
        # Detach gradients to numpy
        grad_x_custom = x_custom.grad
        grad_y_custom = y_custom.grad

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
        
    
    def test_concat_performance(self):
        """
        Test to compare the performance
        """
        
        # Number of iterations for performance test
        n_iters = 10000

        # PyTorch forward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32)
        y = torch.tensor(self.y_raw, dtype=torch.float32)
        for _ in range(n_iters):
            _ = torch.cat([x, y], dim=self.concat_axis)
        t_torch_fwd = time.time() - start

        # Custom forward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32)
        y = Tensor(self.y_raw, dtype=np.float32)
        for _ in range(n_iters):
            _ = Tensor.concat([x, y], axis=self.concat_axis)
        t_custom_fwd = time.time() - start

        # Print performance results
        print(f"torch.cat forward: {t_torch_fwd:.6f}s, concat_forward: {t_custom_fwd:.6f}s")

        # Compute the ratio of custom to torch
        ratio_fwd = t_custom_fwd / t_torch_fwd if t_torch_fwd > 0 else float('inf')
        
        # Assert forward speed is within factor
        self.assertLess(
            ratio_fwd, self.PERFORMANCE_FACTOR,
            msg=(
                f"🟡 Forward kernel too slow: {ratio_fwd:.2f}x slower --> "
                f"torch.cat: {t_torch_fwd:.6f}s, concat_forward: {t_custom_fwd:.6f}s"
            )
        )

        # PyTorch backward
        start = time.time()
        x = torch.tensor(self.x_raw, dtype=torch.float32, requires_grad=True)
        y = torch.tensor(self.y_raw, dtype=torch.float32, requires_grad=True)
        for _ in range(n_iters):
            z = torch.cat([x, y], dim=self.concat_axis)
            z.backward(torch.ones_like(z))
        t_torch_bwd = time.time() - start

        # Custom backward
        start = time.time()
        x = Tensor(self.x_raw, dtype=np.float32, requires_grad=True)
        y = Tensor(self.y_raw, dtype=np.float32, requires_grad=True)
        for _ in range(n_iters):
            out = Tensor.concat([x, y], axis=self.concat_axis)
            out.backward()
        t_custom_bwd = time.time() - start

        # Print performance results
        print(f"torch.cat backward: {t_torch_bwd:.6f}s, concat_backward: {t_custom_bwd:.6f}s")

        # Compute the ratio of custom to torch
        ratio_bwd = t_custom_bwd / t_torch_bwd if t_torch_bwd > 0 else float('inf')
        
        # Assert backward speed is within factor
        self.assertLess(
            ratio_bwd, self.PERFORMANCE_FACTOR,
            msg = (
                f"🟡 Backward kernel too slow: {ratio_bwd:.2f}x slower --> "
                f"torch.cat backward: {t_torch_bwd:.6f}s, concat_backward: {t_custom_bwd:.6f}s"
            )
        )


if __name__ == '__main__':
    unittest.main()