from pettingzoo.classic import tictactoe_v3
from pettingzoo import AECEnv
import numpy as np
import random

# helper functions across baselines
def display_board(board, player_index):
    symbol = ['X', 'O']
    for row in range(3):
        def get_symbol(row, col):
            if board[row, col, 0] == 1:
                return symbol[player_index]
            elif board[row, col, 1] == 1:
                return symbol[1-player_index]
            return " "
        print("|".join([get_symbol(row, col) for col in range(3)]))
        if row < 2:
            print("-----")
    print()

def get_winner(board):
    # Check rows and columns for a win
    for player_index in range(2):
        for i in range(3):
            if np.all(board[:, i, player_index] == 1) or np.all(board[i, :, player_index] == 1):
                return player_index
        # Check diagonals for a win
        if np.all([board[i, i, player_index] == 1 for i in range(3)]) or np.all([board[i, 2 - i, player_index] == 1 for i in range(3)]):
            return player_index
    return -1

def do_move(board, action, player_index, value):
    board[action // 3, action % 3, player_index] = value

class WinBlockAgent:
    def __init__(self, env, name):
        self.observation_space = None
        self.action_space = None
        self.current_observation = None

    def reset(self):
        self.current_observation = None

    def observe(self, observation, reward, termination, truncation, info):
        self.current_observation = observation
        obs_message = f"Observation: {observation}"
        return obs_message

    def act(self):
        # Simple strategy: first try to win, then block, else random
        action = self.find_winning_move(player_index=0)
        if action is None:
            print("searching for block")
            action = self.find_winning_move(player_index=1)
        if action is None:
            print("random move")
            action = self.random_move()
        return action

    def find_winning_move(self, player_index):
        board, action_mask = self.current_observation['observation'], self.current_observation['action_mask']
        for action in range(9):
            if action_mask[action]:
                do_move(board, action, player_index, 1)
                if get_winner(board) == player_index:
                    return action
                do_move(board, action, player_index, 0)
        return None

    def random_move(self):
        action_mask = self.current_observation['action_mask']
        legal_actions = [action for action in range(9) if action_mask[action]]
        return np.random.choice(legal_actions) if legal_actions else None

class RandomAgent:
    def __init__(self, env, name):
        self.action_space = list(range(9))  # Possible actions from 0 to 8
        self.current_observation = None

    def reset(self):
        self.current_observation = None

    def observe(self, observation, reward, termination, truncation, info):
        self.current_observation = observation
        return f"Observation: {observation}, Reward: {reward}"

    def act(self):
        # Implement a strategy to randomly choose from the legal moves
        action_mask = self.current_observation['action_mask']
        legal_actions = [action for action, legal in enumerate(action_mask) if legal]
        if legal_actions:
            return random.choice(legal_actions)
        return None  # In case no legal moves are available



WIN = 1
LOSE = -1
TIE = 0
CONT = 2

## Original
class MiniMaxAgent:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.observation_space = None
        self.action_space = None
        self.current_observation = None

    def reset(self):
        self.current_observation = None

    def observe(self, observation, reward, termination, truncation, info):
        self.current_observation = observation
        obs_message = f"Observation: {observation}"
        return obs_message

    def minimax(self, board, player_index):
        state = self.game_state(board, player_index)
        if state != CONT:
            return -state, None
        
        bestv = -10
        besta = None
        for a in range(9):
            if self.can_move(board, a): 
                do_move(board, a, player_index, 1)
                v, _ = self.minimax(board, 1 - player_index)
                do_move(board, a, player_index, 0)
                if v > bestv:
                    bestv = v
                    besta = a
                if v == 1:
                    return -v, a
        return -bestv, besta
        
    def act(self):
        board = self.current_observation['observation']
        v, action = self.minimax(board, 0)
        print(f"best value is {v}")
        return action
    
    def can_move(self, board, action):
        row, col = action // 3, action % 3
        return board[row, col, 0] == 0 and board[row, col, 1] == 0
    
    # Returns 1 if player_index has won, -1 if opponent has won, 0 if draw, 2 if game is not over
    def game_state(self, board, player_index):
        if get_winner(board) == player_index:
            return WIN
        elif get_winner(board) == 1-player_index:
            return LOSE
        else:
            filled = True
            for a in range(9):
                if self.can_move(board, a):
                    filled = False
                    break
            return TIE if filled else CONT

class HumanAgent:
    def __init__(self):
        self.current_observation = None

    def reset(self):
        self.current_observation = None

    def observe(self, observation, reward, termination, truncation, info):
        self.current_observation = observation
        return f"Observation: {observation}, Reward: {reward}"

    def act(self):
        if self.current_observation is None:
            raise ValueError("No current observation available.")

        # Get the legal moves
        action_mask = self.current_observation['action_mask']

        # Prompt for input until a legal move is entered
        while True:
            try:
                user_input = input("Enter your move (0-8): ")
                action = int(user_input)

                if 0 <= action < 9 and action_mask[action]:
                    return action
                else:
                    print("Illegal move, please try again.")
            except ValueError:
                print("Invalid input. Please enter a number between 0 and 8.")





def simulate(agents: dict[str, any], env: AECEnv) -> dict[any, float]:
    env.reset()
    rewards = {agent: 0 for agent in env.possible_agents}
    
    for agent in agents.values():
        agent.reset()

    for agent_name in env.agent_iter():
        
        observation, reward, termination, truncation, info = env.last()
        display_board(observation['observation'], agent_name=="player_2")
        rewards[agent_name] += reward
        obs_message = agents[agent_name].observe(
            observation, reward, termination, truncation, info
        )
        if termination or truncation:
            action = None
        else:
            action = agents[agent_name].act()
        print(f"Agen {agent_name} Action {action}")
        
        # Display the Visualized board
        env.step(action)
        # display_board(observation['observation'])
    env.close()
    return rewards

# Initialize environment and agents
env = tictactoe_v3.env(render_mode="ansi")
# player_1 is [0,1] > [::1]
# player_2 is [1,0] > [::0]
agents = {'player_1': HumanAgent(), 'player_2': MiniMaxAgent(env, 'player_2')}
agents = {'player_2': HumanAgent(), 'player_1': MiniMaxAgent(env, 'player_1')}

# Simulate the game
rewards = simulate(agents, env)
print("Final Rewards:", rewards)
