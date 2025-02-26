import numpy as np


class NaiveBayes:

    ### Public methods ###

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        - y (np.ndarray): Labels of the dataset
        """

        # Extracting the number of samples and features
        n_samples, n_features = x.shape

        # Extracting the distinct classes
        self.classes = np.unique(y)

        # Extracting the number of distinct classes
        self.n_classes = len(self.classes)

        # Initializing the mean, variance and priors for each class
        self.mean = np.zeros((self.n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((self.n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(self.n_classes, dtype=np.float64)

        # Iterating over the distinct classes
        for idx, c in enumerate(self.classes):
            # Extracting the samples of the current class
            x_c = x[y == c]

            # Computing the mean, variance and prior of the current class
            self.mean[idx, :] = x_c.mean(axis=0)
            self.var[idx, :] = x_c.var(axis=0)
            self.priors[idx] = x_c.shape[0] / float(n_samples)


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the dataset
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        
        Returns:
        - np.ndarray: Predicted labels
        """

        # Computing the posterior probability for each class
        y_pred = [self._predict(sample) for sample in x]

        return np.array(y_pred)


    ### Protected methods ###

    def _predict(self, x: np.ndarray) -> int:
        """
        Predict the label of a single sample
        
        Parameters:
        - x (np.ndarray): Features of the sample
        
        Returns:
        - int: Predicted label
        """

        # Computing the posterior probability for each class
        posteriors = []

        for idx, c in enumerate(self.classes):
            # Computing the prior probability
            prior = np.log(self.priors[idx])

            # Computing the class conditional probability
            conditional = np.sum(np.log(self._pdf(idx, x)))

            # Computing the posterior probability
            posterior = prior + conditional

            # Appending the posterior probability
            posteriors.append(posterior)

        # Returning the class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]
    

    def _pdf(self, idx: int, x: np.ndarray) -> np.ndarray:
        """
        Compute the probability density function
        
        Parameters:
        - idx (int): Index of the class
        - x (np.ndarray): Features of the sample
        
        Returns:
        - np.ndarray: Probability density function
        """

        # Computing the mean and variance of the current class
        mean = self.mean[idx]
        var = self.var[idx]

        # Computing the probability density function
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator