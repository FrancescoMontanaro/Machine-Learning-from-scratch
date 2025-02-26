import warnings
import numpy as np

# Ignoring the warnings
warnings.filterwarnings('ignore')


class LogisticRegression:

    ### Magic methods ###

    def __init__(self, alpha: float = 1e-3, epochs: int = 1000, lambda_reg: float = 1.0) -> None:
        """
        Class constructor
        
        Parameters:
        - alpha (float): Learning rate
        - epochs (int): Number of iterations
        - lambda_reg (float): Regularization term
        """
        
        self.alpha = alpha
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None


    ### Public methods ###

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Function that fits the model to the dataset
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        - y (np.ndarray): Labels of the dataset
        """
        
        # Extracting the number of samples and features
        n_samples, n_features = x.shape

        # Initializing the weights and the bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Iterating over the specified number of epochs
        for _ in range(self.epochs):
            # Computing the linear predictions
            linear_predictions = np.dot(x, self.weights) + self.bias

            # Computing the sigmoid predictions
            y_pred = self._sigmoid(linear_predictions)

            # Computing the Ridge Regression Term
            ridge_w = (2 * self.lambda_reg / n_samples) * self.weights
            ridge_b = (2 * self.lambda_reg / n_samples) * self.bias

            # Computing the gradients with regularization term
            dw = (1 / n_samples) * np.dot(x.T, (y_pred - y)) + ridge_w
            db = (1 / n_samples) * np.sum(y_pred - y) + ridge_b

            # Updating the weights and the bias
            self.weights -= self.alpha * dw
            self.bias -= self.alpha * db


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Function that computes the predictions
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        
        Returns:
        - np.ndarray: Predicted labels
        """
        
        # Checking if the model is trained
        if self.weights is None:
            raise Exception("Fit the model first!")
        
        # Computing the linear predictions
        linear_predictions = np.dot(x, self.weights) + self.bias

        # Computing the sigmoid predictions
        y_pred = self._sigmoid(linear_predictions)

        # Converting the predictions to binary values
        class_predictions = [0 if y <= 0.5 else 1 for y in y_pred]

        # Returning the predictions
        return np.array(class_predictions)


    ### Protected methods ###

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Method that computes the sigmoid function of a given input
        
        Parameters:
        - x (np.ndarray): Input values
        
        Returns:
        - np.ndarray: Output values
        """
        
        # Compute the sigmoid function
        return 1 / (1 + np.exp(-x))