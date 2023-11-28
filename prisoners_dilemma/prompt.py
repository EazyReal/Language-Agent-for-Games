from prompts.default_agent import default_starter

info = '''
# Prisoner's Dilemma

| Import             | `import prisoners_dilemma` |
|--------------------|--------------------------------------------|
| Actions            | Discrete                                   |
| Parallel API       | Yes                                        |
| Manual Control     | No                                         |
| Agents             | `agents= ['player_0', 'player_1']`         |
| Agents             | 2                                          |
| Action Shape       | Discrete(2)                                |
| Action Values      | Discrete(2)                                |
| Observation Shape  | Discrete(2)                                |
| Observation Values | Discrete(2)                                |


Prisoner's Dilemma is a classic game in game theory where two individuals are arrested, and each must decide whether to cooperate with or betray the other.
The game is played in discrete rounds, and each player chooses between two actions: "COOPERATE" (encoded as 0) or "BETRAY" (encoded as 1)
The rewards are determined based on the joint actions of the players.

### Arguments

```python
prisoners_dilemma.env(max_cycles=15)
```

`max_cycles`:  after max_cycles steps all agents will return done.

### Observation Space
The observation space is a scalar value representing the action of the other player in the previous round.
The space is discrete with 2 possible values:

| Value	| Observation |
| -----	| ----------- |
| 0	    | COOPERATE   |
| 1	    | BETRAY   |

### Action Space
The action space is a scalar value with 2 possible values, representing the player's choice of action:
| Value	| Action |
| -----	| ----------- |
| 0	    | COOPERATE   |
| 1	    | BETRAY   |
'''.strip()

starter = default_starter.replace('game_name_v0', 'prinsoners_dilemma_v0')
