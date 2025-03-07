"""
Tools package for DeepSeek R1 LangGraph Agent
"""
from typing import List, Callable, Dict, Any
from langchain_core.tools import BaseTool

# Import all tools
from src.tools.computation import custom_computation
from src.tools.moon_weather import moon_weather

# Create a dictionary of tools with their prompt instructions
from src.tools.computation import get_prompt_template as get_computation_prompt
from src.tools.moon_weather import get_prompt_template as get_moon_weather_prompt

# Export all tools
__all__ = ["custom_computation", "moon_weather",
           "get_all_tools", "get_combined_prompt_template"]


def get_all_tools() -> List[BaseTool]:
    """
    Get all available tools.

    Returns:
        List of all available tools
    """
    return [custom_computation, moon_weather]


def get_tool_prompts() -> Dict[str, Callable[[], str]]:
    """
    Get a dictionary mapping tool names to functions that return their prompt templates.

    Returns:
        Dictionary mapping tool names to prompt template functions
    """
    return {
        "custom_computation": get_computation_prompt,
        "moon_weather": get_moon_weather_prompt
    }


def get_combined_prompt_template() -> str:
    """
    Combines all tool-specific prompt templates into a single template.

    Returns:
        Combined prompt template
    """
    tool_prompts = get_tool_prompts()
    combined_prompt = ""

    # Add prompt instructions for each tool
    for tool_name, get_prompt in tool_prompts.items():
        combined_prompt += get_prompt()
        combined_prompt += "\n\n"

    return combined_prompt
