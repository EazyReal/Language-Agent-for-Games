from configs import EnvConfig
from prompts.prompt import get_solution_prompt, simulation_code
from TicTacToe.prompt import tictactoe_starter, tictactoe_info
from pettingzoo.classic import tictactoe_v3
from pettingzoo import AECEnv
import TicTacToe.baselines

prompt_get_agent_class = prompt_get_agent_class = get_solution_prompt\
        .replace("<simulation code>", simulation_code)\
        .replace("<starter>", tictactoe_starter)\
        .replace("<env info>", tictactoe_info)
env_config = EnvConfig(prompt_get_agent_class, tictactoe_v3)