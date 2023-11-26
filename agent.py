# Agent class represents the state of the agent,
# including the history observations, actions, and the current state.
# it also has methods to interact with the environment.
class Agent:
    def __init__(self, env):
        self.env = env
        self.observations = []
        self.actions = []

    # Here are the admissible actions the agent can take:

    # Place a white/gray/black stone on the board
    def place(self, position):
        self.env.step(position)
        self.observations.append(self.env.get_observation())
        self.actions.append(position)
