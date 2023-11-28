# from abc import ABCMeta, abstractmethod
from typing import Union, Literal, Callable, Dict
from pettingzoo import AECEnv
from dataclasses import dataclass
from pathlib import Path

@dataclass
class EnvConfig:
    prompt_get_agent_class: str
    get_environment: Callable[..., AECEnv]
    baselines: Dict[str, type]

@dataclass
class LMConfig:
    gpt_model: Literal["GPT3.5", "GPT4"]
    max_tokens: int
    log_path: Path
    log_file: Path