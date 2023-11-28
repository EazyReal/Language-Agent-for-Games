default_starter = \
"""
# This is the starting point of your agent class, you should not change the existing method signatures.
# However, you can add more methods or change the implementations to obtain a better strategy.
class Agent:
    import random
    # You can initialize the agent's state here
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.history = {
            "observations": [],
            "actions": []
        }

    # You can reset the agent's state here
    def reset(self):
        \"""
        Reset the agent's state.
        \"""
        self.history = {
            "observations": [],
            "actions": []
        }
    
    # You can update the agent's state
    # This is an example of storing the observation.
    def observe(self, observation, reward, termination, truncation, info):
        self.history["observations"].append(observation)
    
    # You should return the action you want to take and write a docstring to explain why it is a good action.
    # This is an example of choosing the action randomly.
    # self.env.action_space is a method, not a subscriptable object.
    def act(self):
        action = self.env.action_space(self.name).sample()
        self.history["actions"].append(action)
        return action
""".strip()


dummy_agent_code = \
"""
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
        action = self.env.action_space(self.name).sample()
        self.history["actions"].append(action)
        return action
    """