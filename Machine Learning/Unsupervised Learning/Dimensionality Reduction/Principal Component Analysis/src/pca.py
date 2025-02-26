import numpy as np

class PCA:

    ### Magic methods ###

    def __init__(self, n_components: int) -> None:
        """
        Class constructor
        
        Parameters:
        - n_components (int): Number of principal components
        """

        self.n_components = n_components
        self.mean = None

    
    ### Public methods ###

    def fit(self, x: np.ndarray) -> None:
        """
        Fit the model
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        """

        # Centering the dataset
        self.mean = np.mean(x, axis=0)
        x = x - self.mean

        # Computing the covariance matrix
        cov = np.cov(x.T)

        # Computing the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Transposing the eigenvectors
        eigenvectors = eigenvectors.T

        # Sorting the eigenvectors and eigenvalues
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[indices]

        # Selecting the first n components
        self.components = eigenvectors[:self.n_components]

    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform the dataset by reducing its dimensionality projecting it to a lower dimensional subspace defined by the principal components
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        
        Returns:
        - np.ndarray: Transformed dataset
        """
        
        # Check if the mean is a valid vector
        if self.mean is None:
            raise Exception("Fit the model first!")
        
        # Centering the dataset
        x = x - self.mean

        # Projecting the dataset to the lower dimensional subspace
        return np.dot(x, self.components.T)