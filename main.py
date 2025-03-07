#!/usr/bin/env python3
"""
DeepSeek R1 with LangGraph - A simplified agent implementation
"""
import argparse
from simple_langgraph import run_agent


def main():
    """Main entrypoint for the application"""
    parser = argparse.ArgumentParser(
        description='Run DeepSeek agent with LangGraph',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Make query a positional argument that's optional with default value
    parser.add_argument(
        'query',
        nargs='?',  # Makes it optional
        default='Explain the significance of reinforcement learning in AI.',
        help='Query to run'
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the agent
    print(f"Running query: {args.query}")
    response = run_agent(args.query)
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
