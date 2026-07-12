import logging

import ollama

from config import Config

logger = logging.getLogger(__name__)


class OllamaClient(object):
    def __init__(self, config: Config):
        self.config = config

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            response = ollama.chat(
                model=self.config.ollama_model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "num_predict": self.config.max_tokens,
                },
            )
            content = response.message.content
            return content.strip() if content else ""
        except Exception:
            logger.exception("Failed to communicate with Ollama.")
        raise
    
    def is_alive(self) -> bool:
        try:
            self.chat("You are a helpful assistant.", "Reply with exactly: OK")
            return True
        except Exception:
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = Config()
    client = OllamaClient(cfg)

    from prompts import build_system_prompt, build_tool_routing_prompt

    test_queries = [
        "Find the man in the red jacket.",
        "How many cars are in the video?",
        "What happens at the 5-minute mark?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        resp = client.chat(build_system_prompt(), build_tool_routing_prompt(query))
        print(resp)