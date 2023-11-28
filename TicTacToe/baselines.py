from pettingzoo.classic import tictactoe_v3
from pettingzoo import AECEnv
import numpy as np
import random


def display_board(board):
    for row in range(3):
        print("|".join(['X' if board[row, col, 0] == 1 else 'O' if board[row, col, 1] == 1 else ' ' for col in range(3)]))
        if row < 2:
            print("-----")
    print()  # Newline for better formatting


class BlockingAgent:
    def __init__(self):
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
        action = self.find_winning_move()
        if action is None:
            action = self.find_blocking_move()
        if action is None:
            action = self.random_move()
        return action

    def find_winning_move(self):
        return self.find_best_move(is_winning_move=True)

    def find_blocking_move(self):
        return self.find_best_move(is_winning_move=False)

    def find_best_move(self, is_winning_move):
        board, action_mask = self.current_observation['observation'], self.current_observation['action_mask']
        for action in range(9):
            if action_mask[action]:
                simulated_board = self.simulate_move(board, action, is_winning_move)
                if self.is_winner(simulated_board, is_winning_move):
                    return action
        return None

    def random_move(self):
        action_mask = self.current_observation['action_mask']
        legal_actions = [action for action in range(9) if action_mask[action]]
        return np.random.choice(legal_actions) if legal_actions else None

    def simulate_move(self, board, action, is_winning_move):
        simulated_board = np.copy(board)
        player_index = 0 if is_winning_move else 1
        row, col = action % 3, action // 3
        simulated_board[row, col, player_index] = 1
        return simulated_board

    def is_winner(self, board, is_winning_move):
        player_index = 0 if is_winning_move else 1
        # Check rows, columns, and diagonals
        for i in range(3):
            if np.all(board[:, i, player_index] == 1) or np.all(board[i, :, player_index] == 1):
                return True
        if np.all([board[i, i, player_index] == 1 for i in range(3)]) or np.all([board[i, 2-i, player_index] == 1 for i in range(3)]):
            return True
        return False



class RandomAgent:
    def __init__(self):
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




class MiniMaxAgent:
    def __init__(self):
        self.observation_space = None
        self.action_space = None
        self.current_observation = None

    def reset(self):
        self.current_observation = None

    def observe(self, observation, reward, termination, truncation, info):
        self.current_observation = observation
        obs_message = f"Observation: {observation}"
        return obs_message

    def minimax(self, board, action_mask, depth, is_maximizing):
        
        # check if game is over
        if self.is_game_over(board):
            return self.evaluate_board(board)

        if is_maximizing:
            max_eval = float('-inf')
            best_action = None
            for action in range(9):
                if action_mask[action]:
                    new_board, new_action_mask = self.simulate_move(board, action_mask, action, True)
                    eval, _ = self.minimax(new_board, new_action_mask, depth-1, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_action = action
            return max_eval, best_action
        else:
            min_eval = float('inf')
            best_action = None
            for action in range(9):
                if action_mask[action]:
                    new_board, new_action_mask = self.simulate_move(board, action_mask, action, False)
                    eval, _ = self.minimax(new_board, new_action_mask, depth-1, True)
                    if eval < min_eval:
                        min_eval = eval
                        best_action = action
            return min_eval, best_action
        
    def act(self):
        board, action_mask = self.current_observation['observation'], self.current_observation['action_mask']
        _, action = self.minimax(board, action_mask, 8, float('-inf'), float('inf'), True)  # Depth set to 3 for example
        return action
    
    
    def is_game_over(self, board):
        # Check for a winner
        for player_index in [0, 1]:
            if self.has_player_won(board, player_index):
                return True

        # Check if the board is full (no more legal moves)
        if not np.any(board[:, :, 0] == 0) and not np.any(board[:, :, 1] == 0):
            return True

        # Otherwise, the game is still ongoing
        return False

    def evaluate_board(self, board):
        # Check if the agent (maximizing player) has won
        if self.has_player_won(board, 0):
            return 1
        # Check if the opponent (minimizing player) has won
        elif self.has_player_won(board, 1):
            return -1
        # Otherwise, it's a draw or ongoing game
        return 0

    
    def has_player_won(self, board, player_index):
        # Check rows and columns for a win
        for i in range(3):
            if np.all(board[:, i, player_index] == 1) or np.all(board[i, :, player_index] == 1):
                return True

        # Check diagonals for a win
        if np.all([board[i, i, player_index] == 1 for i in range(3)]) or np.all([board[i, 2 - i, player_index] == 1 for i in range(3)]):
            return True

        return False

    def simulate_move(self, board, action_mask, action, is_maximizing):
        
        simulated_board = np.copy(board)
        simulated_action_mask = np.copy(action_mask)
        player_index = 0 if is_maximizing else 1
        row, col = action // 3, action % 3
        simulated_board[row, col, player_index] = 1
        simulated_action_mask[action] = 0  # Update the action mask
        return simulated_board, simulated_action_mask



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





def simulate(agents: any, env: AECEnv) -> dict[any, float]:
    env.reset()
    rewards = {agent: 0 for agent in env.possible_agents}
    
    for agent in agents.values():
        agent.reset()

    for agent_name in env.agent_iter():
        
        observation, reward, termination, truncation, info = env.last()
        display_board(observation['observation'])
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
env = tictactoe_v3.env()
agents = {'player_1': HumanAgent(), 'player_2': MiniMaxAgent()}

# Simulate the game
rewards = simulate(agents, env)
print("Final Rewards:", rewards)
