from ..configs import EnvConfig
from .env import PrisonersDilemmaEnv

prompt_get_agent_class = get_solution_prompt()

pd_env_config = EnvConfig(prompt_get_agent_class, PrisonersDilemmaEnv)
