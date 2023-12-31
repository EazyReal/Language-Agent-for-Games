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
 
The main observation is 2 planes of the 3x3 board. For player_1, the first plane represents the placement of Xs, and the second plane shows the placement of Os. The possible values for each cell are 0 or 1; in the first plane, 1 indicates that an X has been placed in that cell, and 0 indicates
that X is not in that cell. Similarly, in the second plane, 1 indicates that an O has been placed in that cell, while 0 indicates that an O has not been placed. For player_2, the observation is the same, but Xs and Os swap positions, so Os are encoded in plane 1 and Xs in plane 2. This allows for
self-play.

 
#### Legal Actions Mask
 
The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.
 
### Action Space
 
Each action from 0 to 8 represents placing either an X or O in the corresponding cell. The cells are indexed as follows:
 
 
```
0 | 3 | 6
_________
 
1 | 4 | 7
_________
 
2 | 5 | 8
```
 
### Rewards
 
| Winner | Loser |
| :----: | :---: |
| +1     | -1    |
 
If the game ends in a draw, both players will receive a reward of 0.
'''.strip()


starter = default_starter.replace('game_name_v0', 'tictactoe_v3')