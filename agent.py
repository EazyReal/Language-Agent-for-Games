from typing import Protocol

class IAgent(Protocol):
    def __init__(self, env, name):
        ...

    def reset(self):
        ...

    def observe(self, observation, reward, termination, truncation, info):
        ...

    def act(self) -> any:
        ...