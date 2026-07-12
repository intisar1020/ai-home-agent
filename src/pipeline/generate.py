"""
generator.py

Main dataset generation pipeline.
"""

import json
import logging
from pathlib import Path

from config import Config
from ollama_client import OllamaClient
from prompts import (
    build_system_prompt,
    build_generation_system_prompt,
    build_tool_routing_prompt,
    build_user_generation_prompt,
)
from validator import ToolCallValidator, QueryValidator, ValidatedToolCool

logger = logging.getLogger(__name__)


class DatasetGenerator:

    def __init__(self, config: Config):
        self.config = config
        self.client = OllamaClient(config)
        self.validator = ToolCallValidator()
        self.query_validator = QueryValidator()

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

    def generate_query(self) -> str:
        for attempt in range(1, self.config.max_retries + 1):
            response = self.client.chat(
                build_generation_system_prompt(),
                build_user_generation_prompt(),
            )
            logger.debug("Raw LLM response (attempt %d): %r", attempt, response)
            try:
                validated = self.query_validator.validate(response)
                return validated.queries[0]
            except Exception as e:
                logger.warning("Query generation attempt %d failed: %s", attempt, e)
        raise RuntimeError("all query generation retries exhausted")

    def generate_tool_call(self, query: str) -> ValidatedToolCool:
        response = self.client.chat(
            build_system_prompt(),
            build_tool_routing_prompt(query),
        )
        return self.validator.validate(response)

    def save_jsonl(self, path: Path, data: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")

    def run(self):
        logger.info("Starting dataset generation...")

        generated = 0

        while generated < self.config.total_samples:
            logger.info("Generating sample %d/%d ...", generated + 1, self.config.total_samples)

            try:
                query = self.generate_query()
            except Exception as e:
                logger.warning("Skipping sample: %s", e)
                continue

            try:
                tool_call = self.generate_tool_call(query)

                sample = {
                    "query": query,
                    "tool": tool_call.tool,
                    "arguments": tool_call.arguments.model_dump(),
                }

                self.save_jsonl(self.config.dataset_file, sample)
                generated += 1

                logger.info("[%d/%d] Accepted", generated, self.config.total_samples)

            except Exception as e:
                logger.warning("Rejected: %s", query)

                self.save_jsonl(
                    self.config.rejected_file,
                    {"query": query, "error": str(e)},
                )

        logger.info("Dataset generation complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    total_samples = Config.total_samples
    config = Config(total_samples=total_samples)
    generator = DatasetGenerator(config)
    generator.run()
