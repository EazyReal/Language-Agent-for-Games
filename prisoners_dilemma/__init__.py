from configs import EnvConfig
from prompts.base import get_solution_prompt, simulation_code
from . import baselines, prompt, environment

prompt_get_agent_class = get_solution_prompt\
        .replace("<simulation code>", simulation_code)\
        .replace("<starter>", prompt.starter)\
        .replace("<env info>", prompt.info)

env_config = EnvConfig(
    prompt_get_agent_class=prompt_get_agent_class,
    get_environment=environment.env,
    baselines={
        "random": baselines.RandomAgent,
        "coop": baselines.AlwaysCooperateAgent,
        "betray": baselines.AlwaysBetrayAgent,
        "last": baselines.CopyLastActionAgent,
        "majority": baselines.CopyMajorityActionAgent,
    }
)
