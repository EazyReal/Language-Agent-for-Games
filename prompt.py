simulation_code = '''
```python
from pettingzoo.classic import tictactoe_v3

# create the environment
env = tictactoe_v3.env()
agents = {}
# create the agent using the Agent code defined by you
agents[env.possible_agents[0]] = Agent(env)
# you will be playing against an opponent agent
agents[env.possible_agents[1]] = OpponentAgent(env, name)


def simulate(agents: any, env: AECEnv) -> dict[any, float]:
    env.reset()
    rewards = {agent: 0 for agent in env.possible_agents}
    
    for agent in agents.values():
        agent.reset()

    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        rewards[agent_name] += reward
        obs_message = agents[agent_name].observe(
            observation, reward, termination, truncation, info
        )
        if termination or truncation:
            action = None
        else:
            action = agents[agent_name].act()
        print(f"Agen {agent_name} Action {action}")
        env.step(action)
    env.close()
    return rewards

# simulate the game
simulate(agents, env)
```
'''.strip()

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
# Agent class represents the state of the agent,
# including the history observations, actions, and the current state.
# it also has methods to interact with the environment.
class Agent:
    def __init__(self, env, name):
        self.name = name
        self.env = env

    # You should use observe() to update the agent's state
    def observe(self, observation, reward, termination, info):
        ...

    # You should use act() to interact with the environment 
    def act(self):
        ...

    # Here are the admissible actions the agent can take:
    # 
    def go_rock(self):
        return 0
'''

# simulaion code, env info, starter to be replaced
basic_info = '''
# You are a given a task to write a game agent to interact with a pettingzoo environment.
Your agent should be able to interact with the environment by calling the following methods:
<simulation code>

Here is the some information regarding the environment:
<env info>

Here is a starter code for the agent:
<starter>
'''.strip()

get_solution_prompt = f'''
{basic_info}
# Now complete the class Agent() to win the game by completing the agent's methods and adding new methods if necessiry to interact with the environment. 
# For each method, you should write a docstring to describe the method.
# For the act() method, you should return the action you want to take and write a docstring to explain why it is a good action.

# You should complete your solution class below and print $$$ before and after your solution class.
'''.strip()

few_shot_prompt = '''
# Here is an example of a successful solution for solving a similar task:
<example>

# Here is the actual task.
<task>
'''.strip()


example = '''
The game is the famous game of rock-paper-scissors. The agent is playing against a random agent. The agent is the first player to play. The agent should win the game. The agent should play rock first. The agent should play paper second. The agent should play scissors third. The agent should win the game.
The solution 
'''.strip()

code_check_prompt = '''
You are given a Python code snippet define a function called solution. 

[Code]
<solution_func>

Question 1: Are there any syntax error present in the code? Answer Yes/No.
Question 2: Fix the syntax errors and output an error-free version of the code. Only Output the revised code after [Revised code] without any other words.
'''.strip()

feedback_fix_prompt = f'''
{basic_info}

# Here is a example of successful solution for solving a similar task:
[Successful example]
receptacles = ['diningtable 1','drawer 2', 'drawer 1', 'sinkbasin 1', 'toilet 1', 'sidetable 2', 'sidetable 1', 'cabinet 1', 'countertop 1', 'microwave 1', 'fridge 1']
agent = Agent(receptacles)
<example>

# Here is the actual task.
# define environment and agent
receptacles = <receptacle_list>
agent = Agent(receptacles)

# <task>
You have generated code of solution() to solve the task. However, you executed the solution() function and get an error message:
<error_msg>

Let's think step by step. Referring to the successful case and the error message, you should complete the solution function with the correct code.
def solution(agent, start_from=1):
'''.strip()

get_start_from_prompt = f'''
Previously, you generated some code defining a solution function as in [Previous solution]. The previous code is executed and outputs some error. Now you just revised the code as in [Revised solution]. Determine from which step these two version differs. You should only output the step number without saying any other words.

[Previous solution]
<previous_solution>

[Revised solution]
<revised_solution>
'''.strip()
