import os
import sys
import torch
import unittest
import numpy as np

# Ensure the src directory is on sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.core.functional.kernel.reductions import sum_1d, reduce_to_shape
from src.tests.base import Test


class TestReductionsKernel(Test):

    def setUp(self):
        """
        Set up the test case.
        """

        # Create random input data
        self.x_raw = np.random.randn(1000).astype(np.float32)


    def test_sum_1d_forward_values(self):
        """
        Test that sum_1d matches torch.sum on forward pass.
        """

        # Custom forward
        out = np.empty((1,), dtype=np.float32)
        sum_1d(self.x_raw, out)

        # PyTorch forward
        expected = torch.sum(torch.tensor(self.x_raw, dtype=torch.float32)).item()

        # Assert values are close
        self.assertTrue(
            np.allclose(out[0], expected, atol=1e-6),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom: {out[0]}\n"
                f"Torch: {expected}"
            )
        )


    def test_reduce_to_shape_forward_values(self):
        """
        Test that reduce_to_shape reduces along broadcasted dimensions.
        """

        x = np.random.randn(2, 3, 4).astype(np.float32)
        target_shape = (2, 1, 4)

        # Custom forward
        out_custom = reduce_to_shape(x, target_shape)

        # PyTorch forward
        out_expected = torch.sum(torch.tensor(x, dtype=torch.float32), dim=1, keepdim=True).numpy()

        # Assert values are close
        self.assertTrue(
            np.allclose(out_custom, out_expected, atol=1e-6),
            msg=(
                f"❌ Forward outputs differ!\n"
                f"Custom: {out_custom}\n"
                f"Torch: {out_expected}"
            )
        )


if __name__ == '__main__':
    unittest.main()
