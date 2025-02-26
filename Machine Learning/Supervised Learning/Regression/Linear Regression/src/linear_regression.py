import numpy as np

class LinearRegression:

    ### Magic methods ###

    def __init__(self, alpha: float = 1e-02, epochs: int = 1000, lambda_reg: float = 1.0) -> None:
        """
        Class constructor
        
        Parameters:
        - alpha (float): Learning rate
        - epochs (int): Number of iterations
        - lambda_reg (float): Regularization term
        """

        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.epochs = epochs
        self.weights = None
        self.bias = None


    ### Public methods ###

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model
        
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
            # Computing the predictions
            y_pred = self.predict(x)

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
        Predict the labels of the dataset
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        
        Returns:
        - np.ndarray: Predicted labels
        """

        # Check if the model is trained
        if self.weights is None:
            raise Exception("Fit the model first!")

        # Predicting the labels
        return np.dot(x, self.weights) + self.bias