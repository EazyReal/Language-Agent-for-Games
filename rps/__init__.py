from configs import EnvConfig
from prompts.prompt import get_solution_prompt, simulation_code
from rps.prompt import rps_starter, rps_info
from pettingzoo.classic.rps.rps import env
import rps.baselines

prompt_get_agent_class = prompt_get_agent_class = get_solution_prompt\
        .replace("<simulation code>", simulation_code)\
        .replace("<starter>", rps_starter)\
        .replace("<env info>", rps_info)
env_config = EnvConfig(prompt_get_agent_class, env)
