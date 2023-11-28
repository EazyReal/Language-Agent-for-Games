import numpy as np
from typing import Optional

class RandomAgent:
    def __init__(self, env, name):
        self.name = name

    def reset(self):
        pass

    def observe(self, observation, reward, termination, truncation, info):
        pass

    def act(self) -> Optional[int]:
        return np.random.choice([0, 1])

class AlwaysCooperateAgent:
    def __init__(self, env, name):
        self.name = name

    def reset(self):
        pass

    def observe(self, observation, reward, termination, truncation, info):
        pass

    def act(self) -> int:
        return 0  # Cooperate

class AlwaysBetrayAgent:
    def __init__(self, env, name):
        self.name = name

    def reset(self):
        pass

    def observe(self, observation, reward, termination, truncation, info):
        pass

    def act(self) -> int:
        return 1  # Betray

class CopyLastActionAgent:
    def __init__(self, env, name):
        self.name = name
        self.last_action = None

    def reset(self):
        self.last_action = None

    def observe(self, observation, reward, termination, truncation, info):
        pass

    def act(self) -> Optional[int]:
        if self.last_action is None:
            self.last_action = np.random.choice([0, 1])
        return self.last_action

class CopyMajorityActionAgent:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.actions_history = []

    def reset(self):
        self.actions_history = []

    def observe(self, observation, reward, termination, truncation, info):
        if observation is not None:
            self.actions_history.append(observation)

    def act(self) -> Optional[int]:
        if not self.actions_history:
            return np.random.choice([0, 1])  # Random choice for the first round

        cooperate_count = np.sum(np.array(self.actions_history) == 0)
        betray_count = np.sum(np.array(self.actions_history) == 1)

        return 0 if cooperate_count >= betray_count else 1  # Cooperate if ties