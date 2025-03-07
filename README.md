# DeepSeek R1 with LangGraph

This project demonstrates how to build an AI agent using the DeepSeek R1 model with the LangGraph framework.

## Prerequisites

- Python 3.8+
- DeepSeek R1 model running locally via Ollama or similar service
- Required packages: langchain-core, langgraph, requests, pydantic

## Installation

```bash
pip install -r requirements.txt
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
```

## Project Structure

- `main.py` - Main entry point, takes a query as argument
- `simple_langgraph.py` - LangGraph implementation using StateGraph
- `tool.py` - Contains custom tools used by the agent

## Architecture

- Uses a custom `DeepSeekLLM` class that implements the LLM interface
- Uses the StateGraph pattern for defining the agent workflow
- Implements custom agent and tool nodes for flexible processing
- Automatically detects calculation queries and uses the appropriate tool

## Troubleshooting

If you encounter any issues:

1. Make sure the DeepSeek R1 model is running locally
2. Check that all dependencies are installed
3. Verify that the model endpoint in the code matches your local setup
4. For recursion errors, you can increase the `max_iterations` parameter in the `run_agent` function

## Requirements

See requirements.txt for the full list of dependencies.
