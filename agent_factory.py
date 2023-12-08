from typing import Protocol
from agent import IAgent
from configs import EnvConfig, LMConfig
from utils import write_to_file, extract_enclosed_text, lm

from prompts.policies import naive_prompt, cot_prompt

class AgentFactory(Protocol):
    def produce_agent(self) -> (type, str):
        ...

    def update(self, history) -> None:
        ...

class DirectPromptAgentFactory(AgentFactory):
    def __init__(self, policy={"<policy>": cot_prompt}):
        self.policy = policy

    def apply_prompt_policy(self, prompt_get_agent_class_env):
        for k, v in self.policy.items():
            prompt_get_agent_class_env = prompt_get_agent_class_env.replace(k, v)
        return prompt_get_agent_class_env

    def produce_agent_class(self, env_config: EnvConfig, lm_config: LMConfig) -> (type, str):
        prompt_get_agent_class = self.apply_prompt_policy(env_config.prompt_get_initial_agent)
        response = lm(prompt_get_agent_class, lm_config)
        define_agent_code = extract_enclosed_text(response, "```python", "```")
        ldict = {}
        exec(define_agent_code, globals(), ldict)
        return ldict["Agent"], define_agent_code

    def update(self, history) -> None:
        return None

class ReflectionAgentFactory(AgentFactory):
    def __init__(self, policy={"<policy>": cot_prompt}):
        self.policy = policy
        self.define_agent_code = ""
        self.hisories = []

    def apply_prompt_policy(self, prompt_get_agent_class_env):
        for k, v in self.policy.items():
            prompt_get_agent_class_env = prompt_get_agent_class_env.replace(k, v)
        if self.hisories:
            prompt_get_agent_class_env = prompt_get_agent_class_env.replace("<history>", self.hisories[-1])
            prompt_get_agent_class_env = prompt_get_agent_class_env.replace("<define_agent_code>", self.define_agent_code)
        return prompt_get_agent_class_env

    def produce_agent_class(self, env_config: EnvConfig, lm_config: LMConfig) -> (type, str):
        if self.hisories:
            prompt_get_agent_class = self.apply_prompt_policy(env_config.prompt_get_reflection_agent)
        else:
            prompt_get_agent_class = self.apply_prompt_policy(env_config.prompt_get_initial_agent)
        response = lm(prompt_get_agent_class, lm_config)
        define_agent_code = extract_enclosed_text(response, "```python", "```")
        self.define_agent_code = define_agent_code
        # define_agent_code = prompts.default_agent.dummy_agent_code
        ldict = {}
        exec(define_agent_code, globals(), ldict)
        return ldict["Agent"], define_agent_code

    def update(self, history) -> None:
        self.hisories.append(history)


class DummyAgentFactory(AgentFactory):
    def __init__(self, agent_class: type):
        self.agent_class = agent_class

    def produce_agent_class(self, env_config, lm_config) -> (type, str):
        return self.agent_class, ""

    def update(self, history) -> None:
        return None