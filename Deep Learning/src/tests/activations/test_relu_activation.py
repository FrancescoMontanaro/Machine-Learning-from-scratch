import os
import sys
import keras
import unittest
import numpy as np
import tensorflow as tf

# Add the project root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.core import Tensor
from src.activations import ReLU


class TestReLUActivation(unittest.TestCase):
    
    def setUp(self) -> None:
        """
        Set up the test case.
        This method will be called before each test.
        """

        # Create random input data
        self.x_np = np.random.randn(3, 5).astype(np.float32)

        # Create tensors for the input data
        self.x_custom_tensor = Tensor(self.x_np, requires_grad=True)
        self.x_torch_tensor = tf.Variable(self.x_np)

        # Instantiate the activation functions
        self.activation_custom = ReLU()
        self.activation_tf = keras.layers.Activation("relu")
        
        
    def test_relu_activation_forward(self) -> None:
        """
        Test to verify that the forward pass of the custom ReLU activation
        """
        
        # Compute the activation values for custom and TensorFlow implementations.
        custom_activation_out = self.activation_custom(self.x_custom_tensor)
        tf_activation_out = self.activation_tf(self.x_torch_tensor)

        # Compare the forward activation values
        self.assertTrue(
            np.allclose(custom_activation_out.data, tf_activation_out.numpy(), atol=1e-5),
            msg=(
                f"❌ Forward loss outputs differ!\n"
                f"Custom Loss: {custom_activation_out.data}\n"
                f"TF Loss: {tf_activation_out.numpy()}"
            )
        )
        
    
    def test_relu_activation_backward(self) -> None:
        """
        Test to verify that the backward pass of the custom ReLU activation
        """
        
        # Compute the activation values for custom and TensorFlow implementations.
        custom_activation_out = self.activation_custom(self.x_custom_tensor)
        custom_activation_out.backward()
        
        # Compute the gradients for the custom implementation
        custom_grad = self.x_custom_tensor.grad
        
        # Compute the gradients for the TensorFlow implementation
        with tf.GradientTape() as tape:
            tf_activation_out = self.activation_tf(self.x_torch_tensor)
        tf_grad = tape.gradient(tf_activation_out, self.x_torch_tensor)
        
        # Check if the gradients are not None
        self.assertIsNotNone(custom_grad, "Custom Tensor grad is None")
        self.assertIsNotNone(tf_grad, "Tf Tensor grad is None")
        
        # Check if the gradients are not None
        if custom_grad is None or tf_grad is None:
            self.fail("Gradients are None!")
        
        # Compare the backward activation values
        self.assertTrue(
            np.allclose(custom_grad, tf_grad, atol=1e-5), # type: ignore
            msg=(
                f"❌ Backward loss outputs differ!\n"
                f"Custom Grad: {custom_grad}\n"
                f"TF Grad: {tf_grad}"
            )
        )
     
   
if __name__ == "__main__":
    unittest.main()