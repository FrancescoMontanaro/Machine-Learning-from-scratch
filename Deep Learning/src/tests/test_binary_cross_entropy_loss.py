import os
import sys
import keras
import unittest
import numpy as np
import tensorflow as tf

# Add the project root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core import Tensor
from src.loss_functions import BinaryCrossEntropy

class TestBCELoss(unittest.TestCase):
    
    def setUp(self) -> None:
        """
        Set up the test case.
        This method will be called before each test.
        """

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
        self.loss_custom = BinaryCrossEntropy(from_logits=False)
        self.loss_tf = keras.losses.BinaryCrossentropy(from_logits = False)
        
        
    def test_bce_loss_forward(self) -> None:
        """
        Test to verify that the forward pass of the custom BinaryCrossEntropy 
        is consistent with TensorFlow's BinaryCrossentropy.
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
    
    
    def test_bce_loss_backward(self) -> None:
        """
        Test to verify that the backward pass (gradient computation) 
        of the custom BinaryCrossEntropy is consistent with TensorFlow's BinaryCrossentropy.
        """
        
        # Compute the gradients for the custom loss
        loss_custom_val = self.loss_custom(self.y_target_tensor, self.y_pred_tensor)
        loss_custom_val.backward()
        
        # Compute the gradients for the TensorFlow loss using GradientTape.
        with tf.GradientTape() as tape:
            loss_tf_val = self.loss_tf(self.y_target_tf, self.y_pred_tf)
        grad_tf = tape.gradient(loss_tf_val, self.y_pred_tf)
        
        # Check that gradients are not None
        self.assertIsNotNone(self.y_pred_tensor.grad, "Custom Tensor grad is None")
        self.assertIsNotNone(grad_tf, "TensorFlow grad is None")
        
        # Check if the gradients are not None
        if self.y_pred_tensor.grad is None or grad_tf is None:
            self.fail("Gradients are None!")
        
        # Compare the gradients for the predictions
        self.assertTrue(
            np.allclose(self.y_pred_tensor.grad, grad_tf.numpy(), atol=1e-5), # type: ignore
            msg=(
                f"❌ Backward gradients differ!\n"
                f"Custom grad:\n{self.y_pred_tensor.grad}\n\n"
                f"TF grad:\n{grad_tf.numpy()}" # type: ignore
            )
        )
        

if __name__ == "__main__":
    unittest.main()