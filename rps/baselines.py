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
