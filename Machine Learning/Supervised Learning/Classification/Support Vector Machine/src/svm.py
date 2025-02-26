import numpy as np
import matplotlib.pyplot as plt


class SVM:

    ### Magic methods ###

    def __init__(self, lr: float = 1e-03, lambda_param: float = 0.01, epochs: int = 1000) -> None:
        """
        Class constructor

        Parameters:
        - lr (float): Learning rate
        - lambda_param (float): Regularization parameter
        - epochs (int): Number of iterations
        """

        # Initialize the hyperparameters
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs

        # Initialize the weights and bias
        self.weights = None
        self.bias = None

    
    ### Public methods ###

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data
        
        Parameters:
        - X (np.ndarray): Features of the dataset
        - y (np.ndarray): Labels of the dataset
        """

        # Extract the number of samples and features
        n_samples, n_features = X.shape

        # Convert the labels to {-1, 1}
        y_ = np.where(y <= 0, -1, 1)

        # Initialize the weights and bias
        self.weights = np.random.rand(n_features)
        self.bias = 0

        # Iterate over the number of epochs
        for _ in range(self.epochs):
            # Iterate over the number of samples
            for idx, x_i in enumerate(X):
                # Checking the condition for updating the weights and bias
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1

                # The condition is satisfied
                if condition:
                    # Updating the weights, the bias is not updated
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    # Updating the weights and bias
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the dataset
        
        Parameters:
        - X (np.ndarray): Features of the dataset
        
        Returns:
        - np.ndarray: Predicted labels
        """

        # Checking if the model has been trained
        if self.weights is None or self.bias is None:
            raise Exception("Train the model first!")
        
        # Computing the linear combination of the weights and features
        linear_model = np.dot(X, self.weights) - self.bias

        # Computing the predicted labels based on the sign of the linear model {1, -1}
        return np.sign(linear_model)
    

    def visualize(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Visualize the model

        Parameters:
        - X (np.ndarray): Features of the dataset
        - y (np.ndarray): Labels of the dataset
        """

        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]
        
        # Create the figure and axis
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=list(y))

        # Get the x-axis limits
        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        # Get the y-axis limits
        x1_1 = get_hyperplane_value(x0_1, self.weights, self.bias, 0)
        x1_2 = get_hyperplane_value(x0_2, self.weights, self.bias, 0)

        # Get the y-axis limits for the negative hyperplane
        x1_1_n = get_hyperplane_value(x0_1, self.weights, self.bias, -1)
        x1_2_n = get_hyperplane_value(x0_2, self.weights, self.bias, -1)

        # Get the y-axis limits for the positive hyperplane
        x1_1_p = get_hyperplane_value(x0_1, self.weights, self.bias, 1)
        x1_2_p = get_hyperplane_value(x0_2, self.weights, self.bias, 1)

        # Plot the hyperplanes
        ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
        ax.plot([x0_1, x0_2], [x1_1_n, x1_2_n], 'k')
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')

        # Setting the labels
        plt.xlabel('X1')
        plt.ylabel('X2')

        # Showing the plot
        plt.show()
