
rps_info = '''
# Rock Paper Scissors
| Import             | `from pettingzoo.classic import rps_v2` |
|--------------------|-----------------------------------------|
| Actions            | Discrete                                |
| Parallel API       | Yes                                     |
| Manual Control     | No                                      |
| Agents             | `agents= ['player_0', 'player_1']`      |
| Agents             | 2                                       |
| Action Shape       | Discrete(3)                             |
| Action Values      | Discrete(3)                             |
| Observation Shape  | Discrete(4)                             |
| Observation Values | Discrete(4)                             |


Rock, Paper, Scissors is a 2-player hand game where each player chooses either rock, paper or scissors and reveals their choices simultaneously. If both players make the same choice, then it is a draw. However, if their choices are different, the winner is determined as follows: rock beats
scissors, scissors beat paper, and paper beats rock.

### Arguments

``` python
rps_v2.env(max_cycles=15)
```
`max_cycles`:  after max_cycles steps all agents will return done.

### Observation Space

#### Rock, Paper, Scissors

If 3 actions are required, the game played is the standard Rock, Paper, Scissors. The observation is the last opponent action and its space is a scalar value with 4 possible values. Since both players reveal their choices at the same time, the observation is None until both players have acted.
Therefore, 3 represents no action taken yet. Rock is represented with 0, paper with 1 and scissors with 2.

| Value  |  Observation |
| :----: | :---------:  |
| 0      | Rock         |
| 1      | Paper        |
| 2      | Scissors     |
| 3      | None         |


### Action Space

#### Rock, Paper, Scissors

The action space is a scalar value with 3 possible values. The values are encoded as follows: Rock is 0, paper is 1 and scissors is 2.

| Value  |  Action |
| :----: | :---------:  |
| 0      | Rock         |
| 1      | Paper        |
| 2      | Scissors     |

### Rewards

| Winner | Loser |
| :----: | :---: |
| +1     | -1    |

If the game ends in a draw, both players will receive a reward of 0.
'''.strip()

rps_starter = '''
class Agent:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.history = {
            "observations": [],
            "actions": []
        }

    def reset(self):
        \"""
        Reset the agent's state.
        \"""
        self.history = {
            "observations": [],
            "actions": []
        }
    
    def observe(self, observation, reward, termination, truncation, info):
        \"""
        Update the agent's state by storing the observation in the history.
        \"""
        self.history["observations"].append(observation)
    
    def act(self):
        \"""
        Choose the action to take based on the current observation.
        In this game, we will choose the action randomly.
        \"""
        # this is a example of choosing the action randomly, you may want to change it.
        action = self.env.action_space(self.name).sample()
        self.history["actions"].append(action)
        return action
'''
