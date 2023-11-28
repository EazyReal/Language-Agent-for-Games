from pettingzoo.classic import tictactoe_v3
from pettingzoo import AECEnv
import numpy as np
import random

# 
def display_board(board):
    for row in range(3):
        print("|".join(['X' if board[row, col, 0] == 1 else 'O' if board[row, col, 1] == 1 else ' ' for col in range(3)]))
        if row < 2:
            print("-----")
    print()  # Newline for better formatting


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





class MinimaxAgent:
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
        # Use minimax to find the best move
        action = self.minimax_decision()
        return action

    def minimax_decision(self):
        best_score = -float('inf')
        best_action = None
        action_mask = self.current_observation['action_mask']
        board = self.current_observation['observation']

        for action in range(9):
            if action_mask[action]:
                simulated_board = self.simulate_move(board, action, True)
                score = self.min_value(simulated_board)
                if score > best_score:
                    best_score = score
                    best_action = action

        return best_action

    def min_value(self, board):
        if self.is_terminal(board):
            return self.evaluate(board)
        v = float('inf')
        for action in range(9):
            if self.is_legal_move(board, action):
                v = min(v, self.max_value(self.simulate_move(board, action, False)))
        return v

    def max_value(self, board):
        if self.is_terminal(board):
            return self.evaluate(board)
        v = -float('inf')
        for action in range(9):
            if self.is_legal_move(board, action):
                v = max(v, self.min_value(self.simulate_move(board, action, True)))
        return v

    def is_legal_move(self, board, action):
        row, col = action % 3, action // 3
        return board[row, col, 0] == 0 and board[row, col, 1] == 0

    def simulate_move(self, board, action, is_max_player):
        simulated_board = np.copy(board)
        player_index = 0 if is_max_player else 1
        row, col = action % 3, action // 3
        simulated_board[row, col, player_index] = 1
        return simulated_board

    def is_terminal(self, board):
        # Game is terminal if it is won or there are no legal moves left
        return self.is_winner(board, True) or self.is_winner(board, False) or all(self.is_legal_move(board, action) for action in range(9))

    def evaluate(self, board):
        # This function should return a score for the board
        # Positive score for wins, negative for losses, 0 for a draw
        if self.is_winner(board, True):
            return 1  # Max player wins
        elif self.is_winner(board, False):
            return -1  # Min player wins
        else:
            return 0  # Draw

    def is_winner(self, board, is_max_player):
        player_index = 0 if is_max_player else 1
        # Check rows, columns, and diagonals
        for i in range(3):
            if np.all(board[:, i, player_index] == 1) or np.all(board[i, :, player_index] == 1):
                return True
        if np.all([board[i, i, player_index] == 1 for i in range(3)]) or np.all([board[i, 2-i, player_index] == 1 for i in range(3)]):
            return True
        return False




# class myMinMaxAgent:
#     def __init__(self):
#         self.observation_space = None
#         self.action_space = None
#         self.current_observation = None

#     def reset(self):
#         self.current_observation = None

#     def observe(self, observation, reward, termination, truncation, info):
#         self.current_observation = observation
#         obs_message = f"Observation: {observation}"
#         return obs_message
    
    
#     # def minimax(self, board, depth, is_maximizing):
#     #     if depth == 0 or self.is_game_over(board):
#     #         return self.evaluate_board(board), None

#     #     if is_maximizing:
#     #         max_eval = float('-inf')
#     #         best_action = None
#     #         for i in range(3):
#     #             for j in range(3):
#     #                 if (board[i, j, 0] == 0) and (board[i, j, 1] == 0) :
#     #                     board[i, j, 0] = 1  # Assuming 'X' is the AI player
#     #                     eval, _ = self.minimax(board, depth-1, False)
#     #                     board[i, j, 0] = 0
#     #                     if eval > max_eval:
#     #                         max_eval = eval
#     #                         # best_action = (i, j)
#     #                         best_action = i*3 + j
#     #         return max_eval, best_action
#     #     else:
#     #         min_eval = float('inf')
#     #         best_action = None
#     #         for i in range(3):
#     #             for j in range(3):
#     #                 if (board[i, j, 0] == 0) and (board[i, j, 1] == 0) :
#     #                     board[i, j, 1] = 0  # Assuming 'O' is the opponent
#     #                     eval, _ = self.minimax(board, depth-1, True)
#     #                     board[i, j, 1] = 0
#     #                     if eval < min_eval:
#     #                         # best_action = (i, j)
#     #                         best_action = i*3 + j
#     #         return min_eval, best_action
        
        

#     def minimax(self, board, action_mask, depth, is_maximizing):
#         if depth == 0 or self.is_game_over(board):
#             return self.evaluate_board(board), None

#         if is_maximizing:
#             max_eval = float('-inf')
#             best_action = None
#             for action in range(9):
#                 if action_mask[action]:
#                     new_board, new_action_mask = self.simulate_move(board, action_mask, action, True)
#                     eval, _ = self.minimax(new_board, new_action_mask, depth-1, False)
#                     if eval > max_eval:
#                         max_eval = eval
#                         best_action = action
#             return max_eval, best_action
#         else:
#             min_eval = float('inf')
#             best_action = None
#             for action in range(9):
#                 if action_mask[action]:
#                     new_board, new_action_mask = self.simulate_move(board, action_mask, action, False)
#                     eval, _ = self.minimax(new_board, new_action_mask, depth-1, True)
#                     if eval < min_eval:
#                         min_eval = eval
#                         best_action = action

#             return min_eval, best_action

#     def act(self):
#         board, action_mask = self.current_observation['observation'], self.current_observation['action_mask']
#         print(board[0,0,0])
#         # _, action = self.minimax(board, action_mask, 8, True)  # Depth set to 3 for example
#         _, action = self.minimax(board, 8, True)  # Depth set to 3 for example
#         return action
    
#     def is_game_over(self, board):
#         # Check for a winner
#         for player_index in [0, 1]:
#             for i in range(3):
#                 if np.all(board[:, i, player_index] == 1) or np.all(board[i, :, player_index] == 1):
#                     return True
#             if np.all([board[i, i, player_index] == 1 for i in range(3)]) or np.all([board[i, 2-i, player_index] == 1 for i in range(3)]):
#                 return True
#         # Check if the board is full (simplified)
#         if np.all(board[:, :, 0] + board[:, :, 1] > 0):
#             # The board is full
#             return True
#         return False

#     def evaluate_board(self, board):
#         # Check if the agent (maximizing player) has won
#         if self.has_player_won(board, 0):
#             return 1
#         # Check if the opponent (minimizing player) has won
#         elif self.has_player_won(board, 1):
#             return -1
#         # Otherwise, it's a draw or ongoing game
#         return 0

#     def has_player_won(self, board, player_index):
#         # Check rows, columns, and diagonals for a win
#         for i in range(3):
#             if np.all(board[:, i, player_index] == 1) or np.all(board[i, :, player_index] == 1):
#                 return True
#         if np.all([board[i, i, player_index] == 1 for i in range(3)]) or np.all([board[i, 2-i, player_index] == 1 for i in range(3)]):
#             return True
#         return False
    
#     # def is_game_over(self, board):
#     #     # Check for a winner or if the board is full
#     #     for player_index in [0, 1]:
#     #         for i in range(3):
#     #             if np.all(board[:, i, player_index] == 1) or np.all(board[i, :, player_index] == 1):
#     #                 return True
#     #         if np.all([board[i, i, player_index] == 1 for i in range(3)]) or np.all([board[i, 2-i, player_index] == 1 for i in range(3)]):
#     #             return True
#     #     if not np.any(board[:,:,0] == 0) and not np.any(board[:,:,1] == 0):
#     #         # The board is full
#     #         return True
#     #     return False

#     # def evaluate_board(self, board):
#     #     # Check if the agent (maximizing player) has won
#     #     if self.has_player_won(board, 0):
#     #         return 1
#     #     # Check if the opponent (minimizing player) has won
#     #     elif self.has_player_won(board, 1):
#     #         return -1
#     #     # Otherwise, it's a draw or ongoing game
#     #     return 0

#     # def has_player_won(self, board, player_index):
#     #     # Check rows, columns, and diagonals for a win
#     #     for i in range(3):
#     #         if np.all(board[:, i, player_index] == 1) or np.all(board[i, :, player_index] == 1):
#     #             return True
#     #     if np.all([board[i, i, player_index] == 1 for i in range(3)]) or np.all([board[i, 2-i, player_index] == 1 for i in range(3)]):
#     #         return True
#     #     return False

#     def simulate_move(self, board, action_mask, action, is_maximizing):
        
#         simulated_board = np.copy(board)
#         simulated_action_mask = np.copy(action_mask)
#         # player_index = 0 if is_maximizing else 1
#         row, col = action // 3, action % 3
#         simulated_board[row, col, player_index] = 1
#         simulated_action_mask[action] = 0  # Update the action mask
#         return simulated_board, simulated_action_mask



# class MiniMaxAgent:
#     def __init__(self):
#         self.observation_space = None
#         self.action_space = None
#         self.current_observation = None

#     def reset(self):
#         self.current_observation = None

#     def observe(self, observation, reward, termination, truncation, info):
#         self.current_observation = observation
#         obs_message = f"Observation: {observation}"
#         return obs_message

#     def minimax(self, board, player_index):
#         # return the best value for the current player, and the action to achieve it
#         if self.game_state(board, player_index) != 2:
#             return -self.game_state(board, player_index), None
#         bestv = -1000
#         besta = None
#         for action in range(9):
#             if self.can_move(board, action): 
#                 self.do_move(board, action, player_index, 1)
#                 v, _ = self.minimax(board, 1 - player_index)
#                 if v > bestv:
#                     bestv = v
#                     besta = action
#                 self.do_move(board, action, player_index, 0)
#         return -bestv, besta
        
#     def act(self):
#         board = self.current_observation['observation']
#         s0 = np.sum(board[:,:,0])
#         s1 = np.sum(board[:,:,1])

#         v, action = self.minimax(board, int(s0 > s1))
#         print(f"agent {int(s0>s1)} best value is {v}")
#         return action
    
#     def can_move(self, board, action):
#         row, col = action % 3, action // 3
#         return board[row, col, 0] == 0 and board[row, col, 1] == 0
    
#     def do_move(self, board, action, player_index, value):
#         row, col = action % 3, action // 3
#         board[row, col, player_index] = value
    
#     # Returns 1 if player_index has won, -1 if opponent has won, 0 if draw, 2 if game is not over
#     def game_state(self, board, player_index):
#         if self.has_player_won(board, player_index):
#             return 1
#         elif self.has_player_won(board, 1-player_index):
#             return -1
#         else:
#             filled = True
#             for a in range(9):
#                 if self.can_move(board, a):
#                     filled = False
#                     break
#             return 0 if filled else 2

#     def has_player_won(self, board, player_index):
#         # Check rows and columns for a win
#         for i in range(3):
#             if np.all(board[:, i, player_index] == 1) or np.all(board[i, :, player_index] == 1):
#                 return True
#         # Check diagonals for a win
#         if np.all([board[i, i, player_index] == 1 for i in range(3)]) or np.all([board[i, 2 - i, player_index] == 1 for i in range(3)]):
#             return True
#         return False



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

# # Initialize environment and agents
# env = tictactoe_v3.env(render_mode="human")
# # player_1 is [0,1] > [::1]
# # player_2 is [1,0] > [::0]
# agents = {'player_1': HumanAgent(), 'player_2': BlockingAgent()}

# # Simulate the game
# rewards = simulate(agents, env)
# print("Final Rewards:", rewards)
