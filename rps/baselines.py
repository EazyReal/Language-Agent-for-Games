import numpy as np
from typing import Optional

class RandomAgent:
    def __init__(self, env, name):
        pass

    def reset(self):
        pass

    def observe(self, observation, reward, termination, truncation, info):
        pass

    def act(self) -> Optional[int]:
        return np.random.choice([0, 1, 2])

class AlwaysRockAgent:
    def __init__(self, env, name):
        pass

    def reset(self):
        pass

    def observe(self, observation, reward, termination, truncation, info):
        pass

    def act(self) -> int:
        return 0  # Rock
    
class CounterOpponentAgent:
    import random
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.opponent_actions = []
    
    # reset the agent's state here
    def reset(self):
        self.opponent_actions = []
    
    # update the agent's state here
    def observe(self, observation, reward, termination, truncation, info):
        self.opponent_actions.append(observation)
 
    def act(self):
        if self.opponent_actions:
            return (self.opponent_actions[-1] + 1) % 3
        return self.random.choice([0, 1, 2])