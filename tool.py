from langchain.tools import tool
from pydantic import BaseModel, Field


class CustomToolInput(BaseModel):
    query: str = Field(description="The computation query")


@tool("custom_computation", args_schema=CustomToolInput, return_direct=True)
def custom_computation(query: str) -> str:
    # Example: a simple arithmetic evaluation
    try:
        result = eval(query)
        return f"The result is {result}."
    except Exception as e:
        return f"Error in computation: {e}"
