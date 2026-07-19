"""
author: intisar chy.
configuration for the llm tool calling dataset generator.
I will update this as project progresses.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    ollama_model: str = "qwen3.5:9b"
    # ollama_base_url: str = "http://localhost:11434/v1"
    # ollama_api_key: str = "ollama"
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 2048

    total_samples: int = 500
    batch_size: int = 5

    max_retries: int = 3

    output_dir: Path = Path("output")
    dataset_file: Path = field(init=False)
    rejected_file: Path = field(init=False)

    log_dir: Path = Path("logs")
    log_file: Path = field(init=False)

    random_seed: int = 42

    def __post_init__(self):
        self.dataset_file = self.output_dir / "dataset.jsonl"
        self.rejected_file = self.output_dir / "rejected.jsonl"
        self.log_file = self.log_dir / "generation.log"
