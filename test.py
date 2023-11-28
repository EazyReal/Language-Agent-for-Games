from pathlib import Path
import json
import concurrent.futures
from dotenv import load_dotenv
from typing import Dict, Callable
load_dotenv()

from simulation import run_experiment
from agent_factory import AgentFactory, DirectPromptAgentFactory, ReflectionAgentFactory, DummyAgentFactory
from configs import EnvConfig, LMConfig
import prisoners_dilemma
from stats import get_stats

# in case agent uses the libraries
import numpy as np
import random

log_path_root = Path('./log/test_exp/')

class Agent:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.actions_history = []

    def reset(self):
        self.actions_history = []

    def observe(self, observation, reward, termination, truncation, info):
        if observation is not None:
            self.actions_history.append(observation)

    def act(self):
        if len(self.actions_history) == 0:
            return np.random.choice([0, 1])  # Random choice for the first round

        cooperate_count = np.sum(np.array(self.actions_history) == 0)
        betray_count = np.sum(np.array(self.actions_history) == 1)

        return 0 if cooperate_count >= betray_count else 1  # Cooperate if ties
    
dummy_factory = lambda: DummyAgentFactory(Agent)

get_agent_factories: Dict[str, Callable[..., AgentFactory]] = {
    "dummy": dummy_factory, 
}


def main(get_agent_factories):
    env_config = prisoners_dilemma.env_config
    lm_config = LMConfig(
        gpt_model = 'gpt-3.5-turbo',
        max_tokens=1400,
        log_path=log_path_root,
        log_file=Path('lm_log.txt'),
    )

    # This will collect all results from the experiments
    all_experiment_results = []
    # all_experiment_results_lite = []

    # Using ProcessPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Prepare list of futures
        futures = []
        for agent_factory_name, get_agent_factory in get_agent_factories.items():
            for baseline_name, baseline_class in env_config.baselines.items():
                for go_first in range(2):
                    for id_trial in range(3):
                        # Schedule the execution of each experiment
                        future = executor.submit(
                            run_experiment,
                            agent_factory_name,
                            get_agent_factory,
                            baseline_name,
                            baseline_class,
                            go_first,
                            id_trial,
                            env_config,
                            lm_config,
                            log_path_root / Path("results"),
                        )
                        futures.append(future)

        # Collect results as they are completed
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            all_experiment_results.extend(result)
            # all_experiment_results without game_history and define_agent_code
            # all_experiment_results_lite.append(
            #     {
            #         "agent_factory": result["agent_factory"],
            #         "baseline": result["baseline"],
            #         "go_first": result["go_first"],
            #         "id_trial": result["id_trial"],
            #         "id_iter": result["id_iter"],
            #         "rewards": result["rewards"],
            #     }
            # )
    print(f"done {len(all_experiment_results)} experiments")
    return all_experiment_results

if __name__ == "__main__":
    all_experiment_results = main(get_agent_factories)
    get_stats(all_experiment_results, log_file=Path(log_path_root / Path("stats.json")))