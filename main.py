from pettingzoo import AECEnv
from utils import write_to_file
from pathlib import Path
from agent_factory import AgentFactory, DirectPromptAgentFactory, ReflectionAgentFactory
from configs import *

from dotenv import load_dotenv
load_dotenv()

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

# def math_all(r):
#     reflection_information = []
#     for i in range(r):
#         simulate()
#     return reflection_information 

# def get_baselines():
#     return []

# def run_1_experiment(type_agent_factory: AgentFactory, improvement_rounds=5):
#     agent_factory = type_agent_factory()
#     agent = agent_factory.produce_agent()
#     agents = get_baselines()
#     for i in range(improvement_rounds):
#         reflection_information = math_all(agent, agents)
#         agent_factory.update(reflection_information)
#     return stats

# def main():
#     T = 1
#     for i in range(T):
#         stats = run_1_experiment()

# 'gpt-3.5-turbo'
lm_config = LMConfig(
    gpt_model = 'gpt-3.5-turbo',
    max_tokens=1400,
    log_path=Path('./log/v1/'),
    log_file=Path('lm_log.txt'),
)

def main():
    for _ in range(1):
        policy = {}
        agent_factory: AgentFactory = DirectPromptAgentFactory(policy=policy)
        import rps
        env = rps.env(max_cycles=5, render_mode="human")
        Agent = agent_factory.produce_agent_class(rps.env_config, lm_config)
        agents = {}
        for i, name in enumerate(env.possible_agents):
            agents[name] = Agent(env, name) if i == 0 else rps.baselines.RandomAgent(env, name)
        rewards = simulate(agents, env)
        print(rewards)
    # get_refinement_prompt = f"given the execution result: {info}"
    # refinement_response = 
    # (get_refinement_prompt)

main()