import numpy as np
from collections import Counter

class KNN:

    ### Magic methods ###

    def __init__(self, k: int = 3) -> None:
        """
        Class constructor
        
        Parameters:
        - k (int): Number of neighbours to consider
        """
        
        self.k = k


    ### Public methods ###

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        - y (np.ndarray): Labels of the dataset
        """
        
        # Setting the training data
        self.x_train = x
        self.y_train = y

    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the dataset
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        
        Returns:
        - np.ndarray: Predicted labels
        """

        # Predicting the labels for each sample
        y_pred = [self._predict_sample(sample) for sample in x]

        return np.array(y_pred)


    ### Protected methods ###
    
    def _predict_sample(self, sample: np.ndarray) -> int:
        """
        Predict the label of a sample
        
        Parameters:
        - sample (np.ndarray): Sample to predict
        
        Returns:
        - int: Predicted label
        """

        ### Compute the distances ###

        # Compute the Euclidean distances between the sample and the training data
        distance = [self._euclidean_distance(sample, x) for x in self.x_train]

        ### Get the k nearest neighbours ###

        # Get the indices of the k nearest neighbours
        k_indices = np.argsort(distance)[:self.k]

        ### Determine the majority vote ###

        # Get the labels of the k nearest neighbours
        k_neighbours = [self.y_train[i] for i in k_indices]

        # Count the number of labels
        counter = Counter(k_neighbours)

        # Get the most common label
        return int(counter.most_common(1)[0][0])


    @staticmethod
    def _euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute the euclidean distance between points
        
        Parameters:
        - x1 (np.ndarray): First point
        - x2 (np.ndarray): Second point
        
        Returns:
        - float: Euclidean distance
        """

        # Compute the euclidean distance
        return np.sqrt(np.sum((x1 - x2) ** 2))