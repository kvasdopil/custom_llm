#!/usr/bin/env python3
"""
DeepSeek R1 LangChain Agent Implementation
"""
from typing import ClassVar, Dict, List, Optional, TypedDict, Union, Any
import re
import json
import requests
import numpy as np
from pydantic import Field

# LangGraph imports
from langgraph.graph import END, StateGraph

# LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.llms.base import LLM
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.schema import SystemMessage
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Custom tool import
from src.tools.computation import custom_computation


class DeepSeekLLM(LLM):
    """Wrapper for DeepSeek model."""
    # Define model_version as a proper Pydantic field
    model_version: str = Field(
        default="deepseek-r1:1.5b", description="The version of the DeepSeek model to use")
    name: str = Field(default="deepseek-custom-agent",
                      description="Name of the LLM")

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def __init__(self, model_version: str = "deepseek-r1:1.5b", **kwargs):
        """Initialize DeepSeekLLM with a model version."""
        # Pass all arguments to the parent class
        super().__init__(model_version=model_version, **kwargs)

    def _call(self, prompt: str, stop=None) -> str:
        """Call the DeepSeek model with the given prompt."""
        print(f"Calling DeepSeek model...")

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": self.model_version,
                "messages": [{"role": "user", "content": prompt}],
            },
            stream=True
        )

        print("Streaming response")
        response_text = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    response_json = json.loads(decoded_line)
                    content = response_json.get(
                        "message", {}).get("content", "")
                    response_text += content
                    # Print each part of the message as it arrives
                    print(content, end="", flush=True)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

        print()  # Add a newline after streaming
        return response_text

    @property
    def _llm_type(self) -> str:
        return "custom_deepseek"


# Define helper class for our agent state
class AgentState(TypedDict):
    """State for the agent."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    actions: List[Dict]  # Store tool calls and their results


def create_agent():
    """Create a LangChain agent with tool-calling capabilities."""
    # Set up the model
    # model_version = "deepseek-r1:1.5b"
    model_version = "qwen2.5:1.5b"
    llm = DeepSeekLLM(model_version=model_version)
    # print(f"Using model version: {llm.model_version}")

    # Initialize tools list
    tools = [custom_computation]

    # Create the prompt using LangChain's ChatPromptTemplate
    system_template = """
You are a helpful AI assistant that can use tools to assist users. 
For any mathematical calculation, you MUST use the custom_computation tool. 
DO NOT calculate the result yourself.

You have access to the following tools: {tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

For the custom_computation tool, the action_input should be the mathematical expression directly as a string.
Example for custom_computation: {{"action": "custom_computation", "action_input": "2 + 2"}}

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}
```
"""

    human_template = "{input}\n\n{agent_scratchpad}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])

    # Create the agent using LangChain's structured agent
    agent = create_structured_chat_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Create the agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    return agent_executor


def run_agent(query: str, max_iterations: int = 5):
    """Run the agent with a query.

    Args:
        query: The user's query to process
        max_iterations: Maximum number of iterations to prevent infinite loops

    Returns:
        The agent's response
    """
    agent_executor = create_agent()

    # Run the agent
    try:
        result = agent_executor.invoke(
            {"input": query},
            {"max_iterations": max_iterations}
        )
        return result["output"]
    except Exception as e:
        print(f"Error during agent execution: {e}")
        return f"The agent encountered an error or exceeded the maximum number of iterations. Error: {e}"
