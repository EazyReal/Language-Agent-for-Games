from configs import EnvConfig
from prompts.base import get_initial_agent_code_prompt, get_reflection_agent_code_prompt
from . import baselines, prompt, environment


prompt_get_initial_agent = get_initial_agent_code_prompt\
        .replace("<starter>", prompt.starter)\
        .replace("<env info>", prompt.info)

prompt_get_reflection_agent = get_reflection_agent_code_prompt\
        .replace("<starter>", prompt.starter)\
        .replace("<env info>", prompt.info)

env_config = EnvConfig(
    prompt_get_initial_agent=prompt_get_initial_agent,
    prompt_get_reflection_agent=prompt_get_reflection_agent,
    get_environment=environment.env,
    baselines={
        "random": baselines.RandomAgent,
        "coop": baselines.AlwaysCooperateAgent,
        "betray": baselines.AlwaysBetrayAgent,
        "last": baselines.CopyLastActionAgent,
        "majority": baselines.CopyMajorityActionAgent,
    }
)
