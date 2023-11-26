default_starter = """
# This is the starting point of your agent class, you should not change the existing method signatures.
# However, you can add more methods or change the implementations to obtain a better strategy.
class Agent:
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
    # This is a example of storing the observation.
    def observe(self, observation, reward, termination, truncation, info):
        self.history["observations"].append(observation)
    
    # You should return the action you want to take and write a docstring to explain why it is a good action.
    # This is a example of choosing the action randomly.
    def act(self):
        action = self.env.action_space(self.name).sample()
        self.history["actions"].append(action)
        return action
""".strip()
