import json
import logging
import re

from pydantic import ValidationError, BaseModel
from tools import TOOL_MAP

from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> str:
    text = text.strip()
    block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if block:
        text = block.group(1).strip()
    return text


def _parse_json(text: str) -> dict | list:
    cleaned = _extract_json(text)
    if not cleaned:
        raise ValueError("empty response from LLM")
    logger.debug("parsing JSON: %r", cleaned)
    return json.JSONDecoder(strict=False).decode(cleaned)


@dataclass
class ValidatedToolCool:
    tool: str
    arguments: BaseModel


@dataclass
class ValidatedQuery:
    queries: list[str]


class ToolCallValidator:
    def validate(self, llm_output: str) -> ValidatedToolCool:
        try:
            response = _parse_json(llm_output)
        except Exception as e:
            logger.error("failed to parse LLM output as JSON: %s", e)
            raise ValueError("LLM output is not valid JSON") from e

        if "tool" not in response or "arguments" not in response:
            raise ValueError("LLM output missing required keys: 'tool' and/or 'arguments'")

        tool_name = response["tool"]

        if tool_name not in TOOL_MAP:
            raise ValueError(f"Unknown tool {tool_name}")

        tool = TOOL_MAP[tool_name]

        validated_arguments = tool.schema.model_validate(
            response["arguments"]
        )

        return ValidatedToolCool(tool=tool.name, arguments=validated_arguments)


class QueryValidator:
    def validate(self, llm_output: str) -> ValidatedQuery:
        try:
            data = _parse_json(llm_output)
        except Exception as e:
            logger.error("failed to parse query list as JSON: %s", e)
            raise ValueError("query output is not valid JSON") from e

        if not isinstance(data, list):
            raise ValueError("expected a JSON array of query strings")

        if not all(isinstance(q, str) for q in data):
            raise ValueError("all elements in the query array must be strings")

        return ValidatedQuery(queries=data)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_cases = [
        '{"tool": "detect_person", "arguments": {"camera": "front door", "description": "man in red hoodie"}}',
        '{"tool": "summarize_activity", "arguments": {"camera": "backyard"}}',
        '{"tool": "unknown_tool", "arguments": {}}',
        "not json",
        '{"tool": "detect_person"}',
    ]

    validator = ToolCallValidator()
    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {tc}")
        try:
            result = validator.validate(tc)
            print(f"OK: {result}")
        except Exception as e:
            print(f"ERROR: {e}")