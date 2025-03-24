import os
import sys
import keras
import unittest
import numpy as np
import tensorflow as tf

# Add the project root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core import Tensor
from src.loss_functions import CategoricalCrossEntropy


class TestCCELoss(unittest.TestCase):
    
    def setUp(self) -> None:
        """
        Set up the test case.
        This method will be called before each test.
        """
        
        # Set seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        # Create random prediction and target arrays
        self.y_pred_np = np.random.randn(4, 5).astype(np.float32)
        self.y_target_np = np.random.randn(4, 5).astype(np.float32)

        # Create custom Tensors for predictions and targets
        self.y_pred_tensor = Tensor(self.y_pred_np, requires_grad=True)
        self.y_target_tensor = Tensor(self.y_target_np)

        # Create TensorFlow tensors for predictions and targets.
        self.y_pred_tf = tf.Variable(self.y_pred_np)
        self.y_target_tf = tf.constant(self.y_target_np)

        # Instantiate the loss functions
        self.loss_custom = CategoricalCrossEntropy(from_logits=True)
        self.loss_tf = keras.losses.CategoricalCrossentropy(from_logits=True)
        
        
    def test_cce_loss_forward(self) -> None:
        """
        Test to verify that the forward pass of the custom CategoricalCrossEntropy 
        is consistent with TensorFlow's CategoricalCrossEntropy.
        """
        
        # Compute the loss values for custom and TensorFlow implementations.
        loss_custom_val = self.loss_custom(self.y_target_tensor, self.y_pred_tensor)
        loss_tf_val = self.loss_tf(self.y_target_tf, self.y_pred_tf)

        # Compare the forward loss values
        self.assertTrue(
            np.allclose(loss_custom_val.data, loss_tf_val.numpy(), atol=1e-5), # type: ignore
            msg=(
                f"❌ Forward loss outputs differ!\n"
                f"Custom Loss: {loss_custom_val.data}\n"
                f"TF Loss: {loss_tf_val.numpy()}" # type: ignore
            )
        )
     
   
if __name__ == "__main__":
    unittest.main()