"""
Prompt instructions for the computation tool
"""


def get_prompt_template() -> str:
    """
    Returns the prompt template specifically for the computation tool.

    Returns:
        The prompt template string for the computation tool
    """
    return """
For any mathematical calculation, you MUST use the custom_computation tool. 
DO NOT calculate the result yourself.

For multi-step calculations:
1. First use the custom_computation tool for the first calculation
2. When you receive the result, use the custom_computation tool again with the result in a new calculation

Examples:
1. For the custom_computation tool with a simple calculation:
{{"action": "custom_computation", "action_input": "2 + 2"}}

2. For the custom_computation tool with a multi-step calculation (e.g., "Add 5 and 7, then multiply by 2"):
   First step: {{"action": "custom_computation", "action_input": "5 + 7"}}
   After getting result 12, second step: {{"action": "custom_computation", "action_input": "12 * 2"}}
"""
