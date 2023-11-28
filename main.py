from pettingzoo import AECEnv
from utils import write_to_file
from pathlib import Path
from agent_factory import AgentFactory, DirectPromptAgentFactory, ReflectionAgentFactory
from configs import *
from typing import ModuleType
from agent import IAgent

from dotenv import load_dotenv
load_dotenv()

import rps, prisoners_dilemma

env_config = rps.env_config

lm_config = LMConfig(
    gpt_model = 'gpt-3.5-turbo',
    max_tokens=1400,
    log_path=Path('./log/'),
    log_file=Path('lm_log.txt'),
)

get_agent_factories: Dict[str, Callable[..., AgentFactory]] = {
    "direct": lambda : DirectPromptAgentFactory({}),
   # "reflection": lambda: ReflectionAgentFactory({}),
}

def simulate(agents: any, env: AECEnv, agent_name: str) -> dict[any, float]:
    env.reset()
    game_history = f"Your agent is named {agent_name}.\n"
    rewards = {agent: 0 for agent in env.possible_agents}
    
    for agent in agents.values():
        agent.reset()

    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        rewards[agent_name] += reward
        agents[agent_name].observe(
            observation, reward, termination, truncation, info
        )
        if termination or truncation:
            action = None
        else:
            action = agents[agent_name].act()
        game_history += f"{agent_name} takes action {action}\n"
        env.step(action)
    env.close()
    return rewards, game_history

@dataclass
class Result:
    agent_factory: str
    baseline: str
    go_first: int
    id_trial: int
    id_iter: int
    define_agent_code: str
    rewards: dict[str, float]
    game_history: str

experiment_results = []

# for each agent factory, run 3*2 experiments against all other baselines, each with 2 rounds of iteration that allows the agent factory to improve
for agent_factory_name, get_agent_factory in get_agent_factories.items():
    for baseline_name, baseline_class in env_config.baselines.items():
        for go_first in range(2):
            for id_trial in range(3):
                agent_factory = get_agent_factory()
                for id_iter in range(2):
                    Agent, define_agent_code = agent_factory.produce_agent_class(env_config, lm_config)
                    env = env_config.get_environment()
                    agents = {}
                    for i, name in enumerate(env.possible_agents):
                        if i == go_first:
                            agent_name = name
                            agents[name] = Agent(env, name)
                        else:
                            agents[name] = baseline_class(env, name)
                    rewards, game_history = simulate(agents, env, agent_name)
                    agent_factory.update(game_history)
                    experiment_results.append(
                        Result(
                            agent_factory=agent_factory_name,
                            baseline=baseline_name,
                            go_first=go_first,
                            id_trial=id_trial,
                            id_iter=id_iter,
                            define_agent_code=define_agent_code,
                            rewards=rewards,
                            game_history=game_history
                        )
                    )

# save experiment results
