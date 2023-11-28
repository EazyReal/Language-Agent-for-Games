from configs import EnvConfig
from prompts.prompt import get_solution_prompt, simulation_code
from tictactoe.prompt import tictactoe_starter, tictactoe_info
from pettingzoo.classic.tictactoe.tictactoe import env
from pettingzoo import AECEnv
import tictactoe.baselines

prompt_get_agent_class = get_solution_prompt\
        .replace("<simulation code>", simulation_code)\
        .replace("<starter>", tictactoe_starter)\
        .replace("<env info>", tictactoe_info)
env_config = EnvConfig(prompt_get_agent_class, env)