"""
Prompt instructions for the moon weather tool
"""


def get_prompt_template() -> str:
    """
    Returns the prompt template specifically for the moon weather tool.

    Returns:
        The prompt template string for the moon weather tool
    """
    return """
For questions about weather conditions on the moon, use the moon_weather tool
with the appropriate coordinates.

Example:
For the moon_weather tool:
{{"action": "moon_weather", "action_input": {{"latitude": 40.0, "longitude": 150.0}}}}
"""
