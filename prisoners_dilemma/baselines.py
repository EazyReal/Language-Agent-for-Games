import numpy as np
from typing import Optional

class RandomAgent:
    def __init__(self, name):
        self.name = name

    def reset(self):
        pass

    def observe(self, observation, reward, termination, truncation, info):
        pass

    def act(self) -> Optional[int]:
        return np.random.choice([0, 1])

class AlwaysCooperateAgent:
    def __init__(self, name):
        self.name = name

    def reset(self):
        pass

    def observe(self, observation, reward, done, info):
        pass

    def act(self) -> int:
        return 0  # Cooperate

class AlwaysBetrayAgent:
    def __init__(self, name):
        self.name = name

    def reset(self):
        pass

    def observe(self, observation, reward, done, info):
        pass

    def act(self) -> int:
        return 1  # Betray

class CopyLastActionAgent:
    def __init__(self, name):
        self.name = name
        self.last_action = None

    def reset(self):
        self.last_action = None

    def observe(self, observation, reward, done, info):
        pass

    def act(self) -> Optional[int]:
        return self.last_action

class CopyMajorityActionAgent:
    def __init__(self, name):
        self.name = name
        self.actions_history = []

    def reset(self):
        self.actions_history = []

    def observe(self, observation, reward, done, info):
        if observation is not None:
            self.actions_history.append(observation)

    def act(self) -> Optional[int]:
        if len(self.actions_history) == 0:
            return np.random.choice([0, 1])  # Random choice for the first round

        cooperate_count = np.sum(np.array(self.actions_history) == 0)
        betray_count = np.sum(np.array(self.actions_history) == 1)

        return 0 if cooperate_count >= betray_count else 1  # Cooperate if ties

# Example usage:

agents = {
    "random_agent": RandomAgent("random_agent"),
    "always_cooperate_agent": AlwaysCooperateAgent("always_cooperate_agent"),
    "always_betray_agent": AlwaysBetrayAgent("always_betray_agent"),
    "copy_last_action_agent": CopyLastActionAgent("copy_last_action_agent"),
    "copy_majority_action_agent": CopyMajorityActionAgent("copy_majority_action_agent"),
}

# Simulate
simulate(agents, env)
