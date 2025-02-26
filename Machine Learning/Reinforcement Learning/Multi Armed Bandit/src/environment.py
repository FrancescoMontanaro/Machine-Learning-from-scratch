import numpy as np

class Environment:
    
    ### Magic methods ###

    def __init__(self, n_arms: int, probabilities: np.ndarray | None = None) -> None:
        """
        Class constructor
        
        Parameters:
        - n_arms (int): The number of arms
        - probabilities (np.ndarray): The probabilities of the arms
        """
        
        self.n_arms = n_arms

        # Generating the random probabilities of the arms
        self.probabilities = np.random.rand(n_arms) if probabilities is None else probabilities

    
    ### Public methods ###

    def round(self, pulled_arm: int) -> float:
        """
        Returns the reward from a probability distribution
        
        Parameters:
        - pulled_arm (int): The arm that is pulled
        
        Returns:
        - float: The reward
        """

        # Sampling the reward from a Bernoulli distribution
        return np.random.binomial(1, self.probabilities[pulled_arm])