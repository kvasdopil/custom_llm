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

For questions that require analyzing a calculation result (like checking if a number is even/odd):
1. First use the custom_computation tool to get the result
2. Then analyze the returned number to answer the question

For multi-step calculations:
1. First use the custom_computation tool for the first calculation
2. When you receive the result, use the custom_computation tool again with the result in a new calculation

Examples:
1. For a simple calculation:
{{"action": "custom_computation", "action_input": "2 + 2"}}

2. For checking if a number is even or odd:
{{"action": "custom_computation", "action_input": "123 * 456"}}
After getting the result, analyze if it's even or odd and provide the final answer.

3. For a multi-step calculation:
First step: {{"action": "custom_computation", "action_input": "5 + 7"}}
Second step: {{"action": "custom_computation", "action_input": "12 * 2"}}
"""
