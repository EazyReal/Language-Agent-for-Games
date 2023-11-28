from pettingzoo import AECEnv
from agent_factory import AgentFactory
from configs import EnvConfig, LMConfig
from typing import Callable, Dict, Optional
from agent import IAgent
from pathlib import Path
import json

def simulate(agents: Dict[str, IAgent], env: AECEnv, agent_name: str) -> dict[any, float]:
    env.reset()
    game_history = f"Your agent is named {agent_name}.\n"
    rewards = {agent: 0 for agent in env.possible_agents}
    
    for agent in agents.values():
        agent.reset()

    for agent_key in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        rewards[agent_key] += reward
        agents[agent_key].observe(
            observation, reward, termination, truncation, info
        )
        if termination or truncation:
            action = None
        else:
            try:
                action = agents[agent_key].act()
            except Exception as e:
                raise(f"Error in {agent_key}: {e}")
        game_history += f"{agent_key} takes action {action}\n"
        env.step(action)
    env.close()
    return rewards, game_history

def run_experiment_inner(
        agent_factory_name: str,
        get_agent_factory: Callable[..., AgentFactory],
        baseline_name: str,
        baseline_class: type,
        go_first: int,
        id_trial: int,
        env_config: EnvConfig,
        lm_config: LMConfig,
        log_path: Optional[Path],
    ):
    experiment_results = []
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
            {
                "agent_factory": agent_factory_name,
                "baseline": baseline_name,
                "go_first": go_first,
                "id_trial": id_trial,
                "id_iter": id_iter,
                "define_agent_code": define_agent_code,
                "agent_name": agent_name,
                "rewards": rewards,
            }
        )
    if log_path is not None:
        log_path.mkdir(parents=True, exist_ok=True)
        file_path = Path(log_path / f"{agent_factory_name}_{baseline_name}_{go_first}_{id_trial}.json")
        # file_path_lite = Path(log_path / f"{agent_factory_name}_{baseline_name}_{go_first}_{id_trial}_lite.json")
        with open(file_path, "w+") as fo:
            fo.write(json.dumps(experiment_results))
        # with open(file_path_lite, "w+") as fo:
        #     fo.write(json.dumps(experiment_results_lite))
    return experiment_results

def run_experiment(*args, **kwargs):
    try:
        return run_experiment_inner(*args, **kwargs)
    except Exception as e:
        # Log the exception or handle it as needed
        print(f"Error in experiment: {e}")
        return []