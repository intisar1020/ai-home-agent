"""
tools.py

Defines all available tools for the dataset generator.

Each tool consists of:
    - name
    - description
    - argument schema

This file is the single source of truth for tool definitions.
"""

from dataclasses import dataclass
from typing import Type

from pydantic import BaseModel

from schemas import (
    DetectPersonArgs,
    CheckEntryArgs,
    FindEventArgs,
    CountPeopleArgs,
    SummarizeActivityArgs,
    DetectVehicleArgs,
)


@dataclass(frozen=True)
class Tool:
    """Represents a callable tool."""

    name: str
    description: str
    schema: Type[BaseModel]


TOOLS = [

    Tool(
        name="detect_person",
        description="Search a camera feed for a person matching a description.",
        schema=DetectPersonArgs,
    ),

    Tool(
        name="check_entry",
        description="Check whether a door, window, or gate is open or closed.",
        schema=CheckEntryArgs,
    ),

    Tool(
        name="find_event",
        description="Find the timestamp when a security event occurred.",
        schema=FindEventArgs,
    ),


    Tool(
        name="count_people",
        description="Count how many people are visible in a camera feed.",
        schema=CountPeopleArgs,
    ),

    Tool(
        name="summarize_activity",
        description="Summarize the recent activity captured by a camera.",
        schema=SummarizeActivityArgs,
    ),

    Tool(
        name="detect_vehicle",
        description="Check a camera feed for any vehicles present.",
        schema=DetectVehicleArgs,
    ),
]

TOOL_MAP = {tool.name: tool for tool in TOOLS}

TOOL_NAMES = [tool.name for tool in TOOLS]
