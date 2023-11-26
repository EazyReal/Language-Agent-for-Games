class Agent:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.history = {
            "observations": [],
            "actions": []
        }

    def reset(self):
        \"""
        Reset the agent's state.
        \"""
        self.history = {
            "observations": [],
            "actions": []
        }
    
    def observe(self, observation, reward, termination, truncation, info):
        \"""
        Update the agent's state by storing the observation in the history.
        \"""
        self.history["observations"].append(observation)
    
    def act(self):
        \"""
        Choose the action to take based on the current observation.
        In this game, we will choose the action randomly.
        \"""
        # this is a example of choosing the action randomly, you may want to change it.
        action = self.env.action_space(self.name).sample()
        self.history["actions"].append(action)
        return action