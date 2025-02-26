import numpy as np

class Learner:

    ### Magic methods ###

    def __init__(self, n_arms: int) -> None:
        """
        Class constructor
        
        Parameters:
        - n_arms (int): The number of arms
        """
        
        self.t = 0 # Time
        self.n_arms = n_arms # Number of arms
        self.rewards_per_arm = [[] for _ in range(n_arms)] # Rewards for each arm
        self.collected_rewards = np.array([]) # Collected rewards


    ### Public methods ###

    def updateObservations(self, pulled_arm: int, reward: float) -> None:
        """ 
        Updates the observations for the pulled arm
        
        Parameters:
        - pulled_arm (int): The pulled arm
        - reward (float): The reward
        """
        
        # Updating the rewards for the pulled arm
        self.rewards_per_arm[pulled_arm].append(reward)

        # Update the collected rewards
        self.collected_rewards = np.append(self.collected_rewards, reward)