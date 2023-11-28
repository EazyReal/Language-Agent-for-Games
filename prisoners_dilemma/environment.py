"""
# Prisoner's Dilemma

| Import             | `import prisoners_dilemma` |
|--------------------|--------------------------------------------|
| Actions            | Discrete                                   |
| Parallel API       | Yes                                        |
| Manual Control     | No                                         |
| Agents             | `agents= ['player_0', 'player_1']`         |
| Agents             | 2                                          |
| Action Shape       | Discrete(2)                                |
| Action Values      | Discrete(2)                                |
| Observation Shape  | Discrete(2)                                |
| Observation Values | Discrete(2)                                |


Prisoner's Dilemma is a classic game in game theory where two individuals are arrested, and each must decide whether to cooperate with or betray the other.
The game is played in discrete rounds, and each player chooses between two actions: "COOPERATE" (encoded as 0) or "BETRAY" (encoded as 1)
The rewards are determined based on the joint actions of the players.

### Arguments

```python
prisoners_dilemma.env(max_cycles=15)
```

`max_cycles`:  after max_cycles steps all agents will return done.

### Observation Space
The observation space is a scalar value representing the action of the other player in the previous round.
The space is discrete with 2 possible values:

| Value	| Observation |
| -----	| ----------- |
| 0	    | COOPERATE   |
| 1	    | BETRAY   |

### Action Space
The action space is a scalar value with 2 possible values, representing the player's choice of action:
| Value	| Action |
| -----	| ----------- |
| 0	    | COOPERATE   |
| 1	    | BETRAY   |
"""

import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

# Actions for the agents
COOPERATE = 0
BETRAY = 1

# Rewards for the agents based on their actions
REWARD_MAP = {
    (COOPERATE, COOPERATE): (-1, -1),
    (COOPERATE, BETRAY): (-3, 0),
    (BETRAY, COOPERATE): (0, -3),
    (BETRAY, BETRAY): (-2, -2),
}

def env(**kwargs):
    env = PrisonersDilemmaEnv(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class PrisonersDilemmaEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "prisoners_dilemma"}

    def __init__(
        self,
        max_cycles: int | None = 15,
        render_mode: str | None = None,
    ):
        self.max_cycles = max_cycles
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self._action_spaces = {agent: spaces.Discrete(2) for agent in self.possible_agents}
        self._observation_spaces = {agent: spaces.Discrete(2) for agent in self.possible_agents}
        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Discrete(2)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(2)

    def render(self):
        if self.render_mode is None:
            print("You are calling render method without specifying any render mode.")
            return

        if len(self.agents) == 2:
            string = "Current state: Agent1: {} , Agent2: {}".format(
                ["COOPERATE", "BETRAY"][self.state[self.agents[0]]],
                ["COOPERATE", "BETRAY"][self.state[self.agents[1]]],
            )
        else:
            string = "Game over"
        print(string)
        return string

    def observe(self, agent):
        return self.observations[agent]

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        self._cumulative_rewards[agent] = 0
        self.state[self.agent_selection] = action

        if self._agent_selector.is_last():
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[
                (self.state[self.agents[0]], self.state[self.agents[1]])
            ]

            self.num_moves += 1
            self.truncations = {
                agent: self.num_moves >= self.max_cycles for agent in self.agents
            }

            for i in self.agents:
                self.observations[i] = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]
                ]
        else:
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = None
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()


# Example usage:
def __main__():
    global env
    env = env(max_cycles=2)
    env.reset()
    rewards = {agent: 0 for agent in env.possible_agents}

    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        rewards[agent_name] += reward
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent_name).sample()
        print(f"Agen {agent_name} Action {action}")
        env.step(action)
    env.close()
    print(rewards)

if __name__ == "__main__":
    __main__()