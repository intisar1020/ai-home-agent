"""
prompts.py

make prompts on the fly
"""

from tools import TOOLS


def build_system_prompt() -> str:
    return """
You are a home security video monitoring assistant. Users query saved
security camera footage (not live feeds) to review past events.

Your job is to route the user's request to the correct video-analysis tool.

Rules:

- Choose exactly ONE tool.
- Return ONLY valid JSON.
- Do NOT explain your reasoning.
- Do NOT wrap the output in markdown.
""".strip()


def build_generation_system_prompt() -> str:
    return """
You generate realistic queries about reviewing saved home security
camera footage (not real-time monitoring).

Rules:
- Return ONLY a valid JSON array with a single string.
- Do NOT explain your reasoning.
- Do NOT wrap the output in markdown.
- Use review / investigation language, never "right now" or "current".
""".strip()


def build_user_generation_prompt() -> str:
    return """
Generate ONE user request for querying saved home security camera footage.
This is NOT real-time — the user is reviewing previously recorded video.

Randomly choose from these categories:

— person detection (find someone matching a description in recorded footage)
— entry monitoring (check if a door, window, or gate was opened)
— event search (find when something happened in the recording)
— people counting (count how many people appeared on a camera)
— activity summarization (summarize what happened during a time period)
— vehicle detection (check for cars, trucks, or bikes in the recordings)

Cover both INDOOR and OUTDOOR locations:
- indoor: hallway, living room, kitchen, basement, garage
- outdoor: front porch, backyard, driveway, side gate, fence line, street

Use past-tense or review-oriented language — never "right now" or "currently".

Return ONLY a JSON array with the single query string.

Example:
["Find a person holding delivery goods in my front door"]
""".strip()


def build_tool_routing_prompt(user_query: str) -> str:
    tool_text = ""

    for tool in TOOLS:
        schema = tool.schema.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        tool_text += f"\nTool: {tool.name}\n"
        tool_text += f"Description: {tool.description}\n"

        if len(properties) == 0:
            tool_text += "Arguments: None\n"
        else:
            tool_text += "Arguments:\n"
            for name in required:
                tool_text += f"  - {name}: string\n"

        tool_text += "\n"

    return f"""
Available tools:

{tool_text}
User Request:

"{user_query}"

Select the SINGLE best tool.

Return ONLY valid JSON.

Example:

{{
    "tool": "detect_person",
    "arguments": {{
        "camera": "front door",
        "description": "person in a red hoodie last night"
    }}
}}

Do not explain your answer.
Do not use markdown.
""".strip()
