# Computation Tool

This tool allows the agent to perform mathematical calculations.

## Files

- `__init__.py`: Exports the tool and related functions
- `tool.py`: Contains the implementation of the custom_computation tool
- `prompt.py`: Contains the prompt template with instructions for using this tool

## Usage

The computation tool can handle basic arithmetic operations and many math expressions. Examples include:

- Basic arithmetic: `2+2`, `5*7`, `10/2`
- Exponentiation: `2^3` or `2**3`
- Complex expressions: `(5+3)*2`, `10/2+3`

## Modifying or Extending

To modify the tool's functionality:

1. Edit the implementation in `tool.py`
2. Update the prompt instructions in `prompt.py` if necessary

To extend the computation functionality, you can:

- Add support for more math functions
- Implement error handling for specific types of calculations
- Add type checking or validation for inputs

## Prompt Guidelines

The tool-specific prompt instructions are kept in the `prompt.py` file. This is where you can define:

- When the tool should be used
- Examples of valid inputs
- Multi-step calculation instructions
