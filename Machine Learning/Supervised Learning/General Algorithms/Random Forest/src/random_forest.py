import numpy as np
from tqdm import tqdm
from collections import Counter

from .decision_tree import DecisionTree


class RandomForest:

    ### Magic methods ###

    def __init__(self, n_trees: int = 20, max_depth: int = 100, min_samples_split: int = 2, n_features: int | None = None) -> None:
        """
        Class constructor
        
        Parameters:
        - n_trees (int): Number of trees in the forest
        - max_depth (int): Maximum depth of the trees
        - min_samples_split (int): Minimum number of samples required to split an internal node
        - n_features (int): Number of features to consider when looking for the best split
        """

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []


    ### Public methods ###

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        - y (np.ndarray): Labels of the dataset
        """

        # Initializing the trees list
        self.trees = []

        # Iterating over the number of trees
        for _ in tqdm(range(self.n_trees), desc="Training trees"):
            # Creating the decision tree
            tree = DecisionTree(
                max_depth = self.max_depth, 
                min_samples_split = self.min_samples_split, 
                n_features = self.n_features
            )

            # Creating the bootstrap samples
            x_sample, y_sample = self._bootstrap_samples(x, y)

            # Training the decision tree
            tree.fit(x_sample, y_sample)

            # Adding the tree to the trees list
            self.trees.append(tree)


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the dataset
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        
        Returns:
        - np.ndarray: Predicted labels
        """

        # Check if the trees list is empty
        if len(self.trees) == 0:
            raise Exception("Fit the model first!")

        # Predicting the labels
        y_pred = np.array([tree.predict(x) for tree in self.trees])

        # Computing the majority vote
        y_pred = np.swapaxes(y_pred, 0, 1)
        y_pred = [self._most_common_label(y) for y in y_pred]

        return np.array(y_pred)


    ### Protected methods ###

    def _bootstrap_samples(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Create bootstrap samples of the dataset
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        - y (np.ndarray): Labels of the dataset
        
        Returns:
        - tuple[np.ndarray, np.ndarray]: Bootstrap samples of the dataset
        """
        
        # Extracting the number of samples of the dataset
        n_samples = x.shape[0]

        # Creating the bootstrap indices
        indices = np.random.choice(n_samples, n_samples, replace=True)

        return x[indices], y[indices]
    

    def _most_common_label(self, y: np.ndarray) -> int:
        """
        Find the most common label
        
        Parameters:
        - y (np.ndarray): Labels
        
        Returns:
        - int: Most common label
        """
        
        # Counting the number of occurrences of each label
        collection = Counter(y)

        # Computing the most common label
        most_common = collection.most_common(1)[0][0]

        return most_common