# DeepSeek R1 with LangGraph

This project demonstrates how to build an AI agent using the DeepSeek R1 model with the LangGraph framework.

## Project Structure

```
deepseek-agent/
├── main.py                 # Main entry point
├── src/                    # Source code
│   ├── agent.py            # Core agent implementation
│   ├── llm/                # LLM-related code
│   │   └── __init__.py
│   └── tools/              # Tool implementations
│       ├── __init__.py
│       └── computation.py  # Mathematical computation tool
├── tests/                  # Test files
│   ├── __init__.py
│   ├── run_tests.py        # Test runner
│   └── test_agent.py       # Test cases
└── utils/                  # Utility scripts
    ├── __init__.py
    └── benchmark.py        # Benchmarking utility
```

## Prerequisites

- Python 3.8+
- DeepSeek R1 model running locally via Ollama or similar service
- Required packages: langchain-core, langgraph, requests, pydantic

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For development, install in editable mode:

```bash
pip install -e .
```

## Usage

The project provides a simple agent implementation using LangGraph's StateGraph architecture.

### Running the Application

You can run the application with a query as a positional argument:

```bash
# Basic usage
python main.py "Your question here"

# The query is optional and defaults to a preset question
python main.py
```

### Example Queries

```bash
# Ask a general question
python main.py "What is machine learning?"

# Perform a calculation (will use the custom_computation tool)
python main.py "Calculate 23 * 17"

# Multi-step reasoning (combines tool results)
python main.py "What is 5+7, and then multiply that by 2?"
```

## Running Tests

The project includes unit tests to verify functionality:

```bash
# Run all tests with detailed output
python -m tests.run_tests

# Run tests using the unittest module directly
python -m unittest tests.test_agent

# Run a specific test
python -m unittest tests.test_agent.TestDeepSeekAgent.test_llm_processes_tool_results
```

These tests verify:

- Basic agent responses to general questions
- Detection of simple and complex calculations
- The custom_computation tool functionality
- Extraction of tool calls from responses
- Multi-step reasoning where the agent uses tool results for further processing
- Tool result integration into the agent's reasoning process

## Benchmarking

You can benchmark the agent's performance using the benchmark script:

```bash
# Run with default questions
python -m utils.benchmark

# Use custom questions from a file
python -m utils.benchmark --questions-file my_questions.txt

# Specify output file
python -m utils.benchmark --output-file results.json
```

The benchmark runs the agent on a set of questions and measures:

- Response time for each question
- Success rate (based on response length and tool usage)
- Overall statistics about agent performance

## Architecture

- Uses a custom `DeepSeekLLM` class that implements the LLM interface
- Uses the StateGraph pattern for defining the agent workflow
- Implements custom agent and tool nodes for flexible processing
- Automatically detects calculation queries and uses the appropriate tool
- Supports multi-step reasoning using tool results

## Troubleshooting

If you encounter any issues:

1. Make sure the DeepSeek R1 model is running locally
2. Check that all dependencies are installed
3. Verify that the model endpoint in the code matches your local setup
4. For recursion errors, you can increase the `max_iterations` parameter in the `run_agent` function

## Requirements

See requirements.txt for the full list of dependencies.
