# simulaion code, env info, starter to be replaced
basic_info = '''
You are given a task to write a game agent to interact with a pettingzoo environment.

Here is the some information regarding the environment:
<env info>
'''.strip()

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

get_solution_prompt = f'''
{basic_info}

The code describe how your agent will interact with the environment:
<simulation code>

Here is a starter code to help you get started with coding the agent class,
you can replace the starter code with your own code and add more methods if needed,
but you should not change the method signatures:
<starter>

Now, write the Agent class to interact with the environment and maximize your reward. 
For each method, you should write comments to describe the method.
For the act() method, you should return the action you want to take and write comments to explain why it is a good action.
Your code should be enclosed by ```python and ```.
'''.strip()