# %%
from pettingzoo import AECEnv
from utils import write_to_file
from pathlib import Path
from agent_factory import AgentFactory, DirectPromptAgentFactory, ReflectionAgentFactory, DummyAgentFactory
from configs import *
from simulation import run_experiment
import concurrent.futures
import json

from dotenv import load_dotenv
load_dotenv()

import prisoners_dilemma

env_config = prisoners_dilemma.env_config

lm_config = LMConfig(
    gpt_model = 'gpt-3.5-turbo',
    max_tokens=1400,
    log_path=Path('./log/for/'),
    log_file=Path('lm_log.txt'),
)


import numpy as np
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

result = run_experiment("reflect", dummy_factory, "coop", env_config.baselines["major"], 0, 0, env_config, lm_config, None)
print(result[0]["rewards"], result[1]["rewards"])

all_experiment_results = []
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
                        Path('./log/results/'),
                    )
                    futures.append(future)

    # Collect results as they are completed
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        all_experiment_results.extend(result)

print(len(all_experiment_results))

