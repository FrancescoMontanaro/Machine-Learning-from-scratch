import numpy as np
import matplotlib.pyplot as plt


class KMeans:

    ### Magic methods ###

    def __init__(self, k: int = 5, max_iterations: int = 100, plot_steps: bool = False) -> None:
        """
        Class constructor
        
        Parameters:
        - k (int): Number of clusters
        - max_iterations (int): Maximum number of iterations
        - plot_steps (bool): Plot the steps of the algorithm
        """

        # Initializing the parameters
        self.k = k
        self.max_iterations = max_iterations
        self.plot_steps = plot_steps

        # Creating the list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.k)]

        # The centers (mean feature vector) for each cluster
        self.centroids = []

    
    ### Public methods ###

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the cluster for each sample
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        
        Returns:
        - np.ndarray: Cluster labels
        """

        # Initializing the samples
        self.x = x

        # Extracting the number of features and samples
        self.n_samples, self.n_features = x.shape

        # Initializing the centroids
        random_sample_indices = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = np.array([self.x[i] for i in random_sample_indices])

        # Optimizing the clusters
        for _ in range(self.max_iterations):
            # Assign the sample to the closest cluster
            self.clusters = self._create_clusters(self.centroids)

            # Plotting the steps
            if self.plot_steps:
                self.plot()

            # Computing the new centroids
            old_centroids = self.centroids
            self.centroids = self._compute_centroids(self.clusters)

            # Check if the algorithm has converged
            if self._is_converged(old_centroids, self.centroids):
                break

            # Plotting the steps
            if self.plot_steps:
                self.plot()

        # Classifying the samples
        return self._get_cluster_labels(self.clusters)


    def plot(self) -> None:
        """
        Plot the steps of the algorithm
        """

        # Creating the figure
        fig, ax = plt.subplots()

        # Iterating through the clusters
        for _, index in enumerate(self.clusters):
            # Getting points of the cluster
            point = self.x[index].T

            # Plotting the points
            ax.scatter(*point)

        # Plotting the centroids
        for centroid in self.centroids:
            ax.scatter(*centroid, marker="x", color="black", linewidth=2)

        # Showing the plot
        plt.show()


    ### Protected methods ###

    def _create_clusters(self, centroids: np.ndarray) -> list:
        """
        Create clusters based on the closest centroid
        
        Parameters:
        - centroids (np.ndarray): List of centroids
        
        Returns:
        - list: List of clusters
        """

        # Initializing the list of clusters
        clusters = [[] for _ in range(self.k)]

        # Iterating through the samples
        for index, sample in enumerate(self.x):
            # Getting the closest centroid
            centroid_index = self._closest_centroid(sample, centroids)

            # Assigning the sample to the closest cluster
            clusters[centroid_index].append(index)

        return clusters


    def _compute_centroids(self, clusters: list) -> np.ndarray:
        """
        Compute the centroids of the clusters
        
        Parameters:
        - clusters (list): List of clusters
        
        Returns:
        - np.ndarray: List of centroids
        """

        # Initializing the list of centroids
        centroids = np.zeros((self.k, self.n_features))

        # Iterating through the clusters
        for cluster_index, cluster in enumerate(clusters):
            # Computing the new centroid
            new_centroid = np.mean(self.x[cluster], axis=0)

            # Assigning the new centroid
            centroids[cluster_index] = new_centroid

        return centroids


    def _is_converged(self, old_centroids: np.ndarray, new_centroids: np.ndarray, tol: float = 1e-6) -> bool:
        """
        Check if the algorithm has converged
        
        Parameters:
        - old_centroids (np.ndarray): List of old centroids
        - new_centroids (np.ndarray): List of new centroids
        - tol (float): Tolerance
        
        Returns:
        - bool: True if the algorithm has converged, False otherwise
        """

        # Checking the distance between the old and new centroids
        distances = [self._euclidean_distance(old_centroids[i], new_centroids[i]) for i in range(self.k)]

        return sum(distances) < tol
    

    def _get_cluster_labels(self, clusters: list) -> np.ndarray:
        """
        Get the cluster labels
        
        Parameters:
        - clusters (list): List of clusters
        
        Returns:
        - np.ndarray: Cluster labels
        """

        # Initializing the list of labels
        labels = np.empty(self.n_samples)

        # Iterating through the clusters
        for cluster_index, cluster in enumerate(clusters):
            # Assigning the cluster label to each sample
            for sample_index in cluster:
                labels[sample_index] = cluster_index

        return labels
    

    def _closest_centroid(self, sample: np.ndarray, centroids: np.ndarray) -> int:
        """
        Get the index of the closest centroid
        
        Parameters:
        - sample (np.ndarray): Sample to classify
        - centroids (np.ndarray): List of centroids
        
        Returns:
        - int: Index of the closest centroid
        """

        # Computing the distances
        distances = [self._euclidean_distance(sample, point) for point in centroids]

        # Getting the index of the closest centroid
        closest_index = int(np.argmin(distances))

        return closest_index
    

    @staticmethod
    def _euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute the Euclidean distance
        
        Parameters:
        - x1 (np.ndarray): First point
        - x2 (np.ndarray): Second point
        
        Returns:
        - float: Euclidean distance
        """

        # Computing the distance
        return np.sqrt(np.sum((x1 - x2) ** 2))