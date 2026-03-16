# src/wcr_agent/agent/tool_registry.py

from __future__ import annotations

from wcr_agent.tools.filter_rings_tool import run_filter_rings_tool
from wcr_agent.tools.summarize_rings_tool import run_summarize_rings_tool
from wcr_agent.tools.compare_groups_tool import run_compare_groups_tool


TOOL_REGISTRY = {
    "filter_rings": run_filter_rings_tool,
    "summarize_rings": run_summarize_rings_tool,
    "compare_groups": run_compare_groups_tool,
}


def get_tool(tool_name: str):
    if tool_name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {tool_name}")
    return TOOL_REGISTRY[tool_name]


def list_tools() -> list[str]:
    return sorted(TOOL_REGISTRY.keys())