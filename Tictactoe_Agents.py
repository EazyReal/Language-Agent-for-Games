from pettingzoo.classic import tictactoe_v3
from pettingzoo import AECEnv
import numpy as np

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
        # Implement a simple strategy: Choose the first available legal move
        action_mask = self.current_observation['action_mask']
        for action, legal in enumerate(action_mask):
            if legal:
                return action
        return None  # In case no legal moves are available


class MinMaxAgent:
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

    # def act(self):
    #     board, action_mask = self.current_observation['observation'], self.current_observation['action_mask']
    #     _, action = self.minimax(board, action_mask, True)
    #     return action

    # def minimax(self, board, action_mask, is_maximizing):
    #     if self.is_game_over(board):
    #         return self.evaluate_board(board), None

    #     if is_maximizing:
    #         best_score = float('-inf')
    #         best_action = None
    #         for action in range(9):
    #             if action_mask[action]:
    #                 new_board, new_action_mask = self.simulate_move(board, action_mask, action, True)
    #                 score, _ = self.minimax(new_board, new_action_mask, False)
    #                 if score > best_score:
    #                     best_score = score
    #                     best_action = action
    #         return best_score, best_action
    #     else:
    #         best_score = float('inf')
    #         best_action = None
    #         for action in range(9):
    #             if action_mask[action]:
    #                 new_board, new_action_mask = self.simulate_move(board, action_mask, action, False)
    #                 score, _ = self.minimax(new_board, new_action_mask, True)
    #                 if score < best_score:
    #                     best_score = score
    #                     best_action = action
    #         return best_score, best_action
    
    
    def minimax(self, board, action_mask, depth, alpha, beta, is_maximizing):
        if depth == 0 or self.is_game_over(board):
            return self.evaluate_board(board), None

        if is_maximizing:
            max_eval = float('-inf')
            best_action = None
            for action in range(9):
                if action_mask[action]:
                    new_board, new_action_mask = self.simulate_move(board, action_mask, action, True)
                    eval, _ = self.minimax(new_board, new_action_mask, depth-1, alpha, beta, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_action = action
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval, best_action
        else:
            min_eval = float('inf')
            best_action = None
            for action in range(9):
                if action_mask[action]:
                    new_board, new_action_mask = self.simulate_move(board, action_mask, action, False)
                    eval, _ = self.minimax(new_board, new_action_mask, depth-1, alpha, beta, True)
                    if eval < min_eval:
                        min_eval = eval
                        best_action = action
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return min_eval, best_action

    def act(self):
        board, action_mask = self.current_observation['observation'], self.current_observation['action_mask']
        _, action = self.minimax(board, action_mask, 3, float('-inf'), float('inf'), True)  # Depth set to 3 for example
        return action
    
    
    

    def is_game_over(self, board):
        # Check for a winner or if the board is full
        for player_index in [0, 1]:
            for i in range(3):
                if np.all(board[:, i, player_index] == 1) or np.all(board[i, :, player_index] == 1):
                    return True
            if np.all([board[i, i, player_index] == 1 for i in range(3)]) or np.all([board[i, 2-i, player_index] == 1 for i in range(3)]):
                return True
        if not np.any(board[:,:,0] == 0) and not np.any(board[:,:,1] == 0):
            # The board is full
            return True
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
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if np.all(board[:, i, player_index] == 1) or np.all(board[i, :, player_index] == 1):
                return True
        if np.all([board[i, i, player_index] == 1 for i in range(3)]) or np.all([board[i, 2-i, player_index] == 1 for i in range(3)]):
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




def simulate(agents: any, env: AECEnv) -> dict[any, float]:
    env.reset()
    rewards = {agent: 0 for agent in env.possible_agents}
    
    for agent in agents.values():
        agent.reset()

    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        rewards[agent_name] += reward
        obs_message = agents[agent_name].observe(
            observation, reward, termination, truncation, info
        )
        if termination or truncation:
            action = None
        else:
            action = agents[agent_name].act()
        print(f"Agen {agent_name} Action {action}")
        env.step(action)
    env.close()
    return rewards

# Initialize environment and agents
env = tictactoe_v3.env()
agents = {'player_1': BlockingAgent(), 'player_2': BlockingAgent()}

# Simulate the game
simulate(agents, env)
