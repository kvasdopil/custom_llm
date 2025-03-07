# Agent Tools

This directory contains the tools available to the DeepSeek R1 agent. Each tool is structured as a self-contained module with its own implementation, prompt template, and documentation.

## Structure

The tools module follows a modular design where each tool is contained in its own directory:

```
tools/
├── __init__.py         # Exports all tools and provides helper functions
├── common_prompt.py    # Common prompt instructions for all tools
├── README.md           # This documentation file
├── computation/        # Computation tool module
│   ├── __init__.py     # Exports the computation tool
│   ├── tool.py         # Implementation of the computation tool
│   ├── prompt.py       # Computation-specific prompt instructions
│   └── README.md       # Documentation for the computation tool
└── moon_weather/       # Moon weather tool module
    ├── __init__.py     # Exports the moon weather tool
    ├── tool.py         # Implementation of the moon weather tool
    ├── prompt.py       # Moon weather-specific prompt instructions
    └── README.md       # Documentation for the moon weather tool
```

## Adding a New Tool

To add a new tool to the agent:

1. Create a new directory for your tool: `tools/your_tool_name/`
2. Create the following files:
   - `__init__.py`: Export your tool and get_prompt_template function
   - `tool.py`: Implement your tool functionality
   - `prompt.py`: Define prompt instructions specific to your tool
   - `README.md`: Document your tool's usage and functionality
3. Update the main `tools/__init__.py` file:
   - Import your tool and prompt template function
   - Add your tool to the `get_all_tools()` function
   - Add your prompt template function to the `get_tool_prompts()` function

## Tool Design Guidelines

When creating new tools, follow these guidelines:

1. **Self-Contained**: Each tool should be self-contained with its implementation, prompt instructions, and any helper functions in its own directory
2. **Clear Documentation**: Include a README.md file that explains what the tool does, its inputs/outputs, and usage examples
3. **Specific Instructions**: The prompt.py file should contain clear instructions on when and how to use the tool
4. **Error Handling**: Include appropriate error handling in the tool implementation

## Common Functionality

The `tools/__init__.py` file provides several helper functions:

- `get_all_tools()`: Returns a list of all available tools
- `get_tool_prompts()`: Returns a dictionary mapping tool names to their prompt template functions
- `get_combined_prompt_template()`: Combines all tool-specific prompts into a single template
