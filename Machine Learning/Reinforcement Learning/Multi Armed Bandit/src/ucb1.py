import numpy as np

from .learner import Learner


class UCB1(Learner):
    
    ### Magic methods ###
    
    def __init__(self, n_arms: int) -> None:
        """
        Class constructor
        
        Parameters:
        - n_arms (int): The number of arms
        """

        # Calling the super class constructor
        super().__init__(n_arms)

        # Initializing the empirical means for each arm
        self.empirical_means = np.zeros(n_arms)

        # Initializing the confidence bounds for each arm
        self.confidence = np.array([np.inf] * n_arms)

    
    ### Public methods ###

    def pull_arm(self) -> int:
        """
        Function that pulls an arm
        
        Returns:
        - int: The pulled arm index
        """

        # Computing the upper confidence bounds for each arm
        upper_conf = self.empirical_means + self.confidence

        # Returning the arm with the highest upper confidence bound
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])


    def update(self, pull_arm: int, reward: float) -> None:
        """
        Function that updates the empirical means and the confidence bounds for the pulled arm
        
        Parameters:
        - pull_arm (int): The pulled arm index
        - reward (float): The reward
        """

        # Updating the time
        self.t += 1

        # Updating the empirical means and the confidence bounds for the pulled arm
        self.empirical_means[pull_arm] = (self.empirical_means[pull_arm] * (self.t - 1) + reward) / self.t

        # Iterating over the arms
        for a in range(self.n_arms):
            # Computing and updating the confidence bounds for each arm
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf

        # Updating the observations
        self.updateObservations(pull_arm, reward)