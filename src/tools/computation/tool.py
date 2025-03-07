#!/usr/bin/env python3
"""
Custom computation tool implementation for DeepSeek R1 LangGraph Agent
"""
from langchain.tools import tool
from pydantic import BaseModel, Field


class CustomToolInput(BaseModel):
    """Input schema for the custom_computation tool."""
    query: str = Field(description="The computation query")


@tool("custom_computation", args_schema=CustomToolInput, return_direct=False)
def custom_computation(query: str) -> str:
    """Perform basic arithmetic computation or evaluate simple expressions.

    Args:
        query: A string containing a mathematical expression to evaluate.

    Returns:
        A string containing just the numerical result of the computation.
    """
    # Replace ^ with ** for exponentiation
    query = query.replace("^", "**")

    # Evaluate the mathematical expression
    try:
        result = eval(query)
        return f"The result is {result}."
    except Exception as e:
        return f"Error in computation: {e}"
