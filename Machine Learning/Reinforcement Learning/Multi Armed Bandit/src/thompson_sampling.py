import numpy as np

from .learner import Learner

class ThompsonSampling(Learner):
    
    ### Magic methods ###

    def __init__(self, n_arms: int) -> None:
        """
        Class constructor
        
        Parameters:
        - n_arms (int): The number of arms
        """
        
        # Calling the super class constructor
        super().__init__(n_arms)

        # Initializing the beta parameters for each arm
        self.beta_parameters = np.ones((n_arms, 2))

    
    ### Public methods ###

    def pull_arm(self) -> int:
        """
        Function that pulls an arm
        
        Returns:
        - int: The pulled arm index
        """
        
        # Sampling from the beta distribution the arm with the highest value
        return int(np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1])))


    def update(self, pulled_arm: int, reward: float) -> None:
        """
        Function that updates the beta parameters for the pulled arm
        
        Parameters:
        - pulled_arm (int): The pulled arm index
        - reward (float): The reward
        """

        # Updating the time
        self.t += 1
        
        # Updating the observations
        self.updateObservations(pulled_arm, reward)

        # Updating the beta parameters for the pulled arm
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward