from configs import EnvConfig
from prompts.base import get_solution_prompt, simulation_code
from pettingzoo.classic.rps.rps import env
from . import baselines, prompt

prompt_get_agent_class = get_solution_prompt\
        .replace("<simulation code>", simulation_code)\
        .replace("<starter>", prompt.starter)\
        .replace("<env info>", prompt.info)

env_config = EnvConfig(
    prompt_get_agent_class=prompt_get_agent_class,
    get_environment=env,
    baselines={
        "random": baselines.RandomAgent,
        "always": baselines.AlwaysRockAgent,
        "counter": baselines.CounterOpponentAgent,
    }
)
