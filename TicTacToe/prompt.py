from prompts.default_agent import default_starter

info = '''
 
| Import             | `from pettingzoo.classic import tictactoe_v3` |
|--------------------|-----------------------------------------------|
| Actions            | Discrete                                      |
| Parallel API       | Yes                                           |
| Manual Control     | No                                            |
| Agents             | `agents= ['player_1', 'player_2']`            |
| Agents             | 2                                             |
| Action Shape       | (1)                                           |
| Action Values      | [0, 8]                                        |
| Observation Shape  | (3, 3, 2)                                     |
| Observation Values | [0,1]                                         |
 
 
Tic-tac-toe is a simple turn based strategy game where 2 players, X and O, take turns marking spaces on a 3 x 3 grid. The first player to place 3 of their marks in a horizontal, vertical, or diagonal line is the winner.
 
### Observation Space
The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.
The 'observation' key corresponds to a 3 x 3 x 2 numpy array representing the board.
For the last dimension, the current player taking the action is repsented by 0, and the other player is represented by 1.
For example, board[1, 2, 0] = 1 means that the current player has captured in the middle right cell.

 
#### Legal Actions Mask
 
The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.
 
### Action Space
 
Each action from 0 to 8 represents placing either an X or O in the corresponding cell.
 
### Rewards
 
| Winner | Loser |
| :----: | :---: |
| +1     | -1    |
 
If the game ends in a draw, both players will receive a reward of 0.
'''.strip()


starter = \
"""
# This is the starting point of your agent class, you should not change the existing method signatures.
# However, you can add more methods or change the implementations to obtain a better strategy.
class Agent:
    # import the necessary packages here
    import random
    # initialize the agent's state here
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.observation = None

    # reset the agent's state here
    def reset(self):
        \"""
        Reset the agent's state.
        \"""
        pass
    
    # update the agent's state here
    # This is an example of storing the observation.
    def observe(self, observation, reward, termination, truncation, info):
        self.observation = observation
    
    # return the action here
    # write comments to explain why this is a good policy.
    # this is an example of choosing the action randomly, you may want to change it.
    # self.env.action_space is a method, not a subscriptable object.
    def act(self):
        action = self.env.action_space(self.name).sample()
        return action
    
    # this is a helper function to set a mark on the board at index "action", as player "player", with value "value"
    # `player == 0` means the mark is for the current player taking the action, `player == 1` means the other player.
    def do_move(self, board, action, player, value):
        board[action//3, action%3, player] = value
""".strip()
