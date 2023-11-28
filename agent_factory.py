from typing import Protocol
from agent import IAgent
from configs import EnvConfig, LMConfig
from utils import write_to_file, extract_enclosed_text, lm
import prompts

class AgentFactory(Protocol):
    def produce_agent(self) -> IAgent:
        ...

    def update(self, reflection_information) -> None:
        ...

class DirectPromptAgentFactory(AgentFactory):
    def __init__(self, policy):
        self.policy = policy

    def apply_prompt_policy(self, prompt_get_agent_class_env):
        for k, v in self.policy.items():
            prompt_get_agent_class_env = prompt_get_agent_class_env.replace(k, v)
        return prompt_get_agent_class_env

    def produce_agent_class(self, env_config: EnvConfig, lm_config: LMConfig) -> type:
        prompt_get_agent_class = self.apply_prompt_policy(env_config.prompt_get_agent_class)
        response = lm(prompt_get_agent_class, lm_config)
        define_agent_code = extract_enclosed_text(response, "```python", "```")
        # define_agent_code = prompts.default_agent.dummy_agent_code
        ldict = {}
        exec(define_agent_code, globals(), ldict)
        return ldict["Agent"]

    def update(self, reflection_information) -> None:
        return None

cot_policy = {
    "<policy_1>" : "",
    "<policy_2>" : "think step by step"
}
cot_factory = DirectPromptAgentFactory(cot_policy)

class ReflectionAgentFactory(AgentFactory):
    def __init__(self, policy):
        self.policy

    def produce_agent(self) -> type:
        return None

    def update(self, reflection_information) -> None:
        pass