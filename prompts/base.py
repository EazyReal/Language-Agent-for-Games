simulation_code = '''
```python
from pettingzoo.classic import game_name_v0

# create the environment
env = game_name_v0.env()

# create your agent and the opponent agent
agents = {
    env.possible_agents[0]: Agent(env, env.possible_agents[0]),
    env.possible_agents[1]: OpponentAgent(env, env.possible_agents[1])
}

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

simulate(agents, env)
```
'''.strip()

task_info = f"""
You are given a task to write a game agent to interact with a pettingzoo environment.

The environment is defined as follows:
<env info>

The code describe how your agent will interact with the environment:
{simulation_code}
"""

# replace <env info> <starter> <policy>
get_initial_agent_code_prompt = f'''
{task_info}

Here is a starter code to help you get started with coding the agent class,
you can replace the starter code with your own code and add more methods if needed,
but you should not change the method signatures:
<starter>

<policy>
'''.strip()

# replace <env info> <starter> <define_agent_code> <history>
get_reflection_agent_code_prompt = f'''
{task_info}

You have written the following code to play the game with another agent:
```python
<define_agent_code>
```

And here is the result of your agent playing against another agent:
<history>

Now, based on your last agent, the rewards it obtained, and the history, write a new Agent class.
First, think about what you have done and what you can do to improve the agent's performance.
- Recap some basic game theory knowledge (Nash equilibrium, best response) and apply it to the situation.
- Think about what the other agent is doing and how you can exploit it.
Second, write a new Agent class to interact with the environment and the agent you interacted with and maximize your reward. 
For the coding part:
- Enclose the code by ```python and ```
- Only write the new Agent class, write comments to explain your reasoning.
- `import` needs to be under the Agent scope. You can only import from the standard library and numpy.

The response format should be:
[Thoughts] <your thoughts>
[Code]
```python
class Agent:
    ... # your code
```
'''.strip()