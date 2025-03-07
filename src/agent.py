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
from src.tools.moon_weather import moon_weather


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

        # Clean response by removing extra markdown-style code blocks
        # This helps with JSON parsing if the LLM adds formatting
        response_text = self.clean_response(response_text)

        return response_text

    def clean_response(self, text: str) -> str:
        """Clean LLM response to handle common formatting issues.

        Args:
            text: Raw text response from the LLM

        Returns:
            Cleaned text that's better suited for parsing
        """
        # Remove markdown-style code blocks that might interfere with JSON parsing
        text = re.sub(r'```(?:json|json_blob)?', '', text)
        text = re.sub(r'```', '', text)

        # Remove extra whitespace and indentation that might break parsing
        text = text.strip()

        return text

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
    tools = [custom_computation, moon_weather]

    # Create the prompt using LangChain's ChatPromptTemplate
    system_template = """
You are a helpful AI assistant that can use tools to assist users. 

For ANY general knowledge questions that don't involve calculations or moon weather, simply respond directly with:
{{"action": "Final Answer", "action_input": "Your detailed answer here"}}

IMPORTANT: Action names MUST be capitalized exactly as shown:
- "Final Answer" (not "final_answer" or "FINAL ANSWER")
- "custom_computation" (not "Custom_Computation" or "CUSTOM_COMPUTATION")
- "moon_weather" (not "Moon_Weather" or "MOON_WEATHER")

For any mathematical calculation, you MUST use the custom_computation tool. 
DO NOT calculate the result yourself.

For multi-step calculations:
1. First use the custom_computation tool for the first calculation
2. When you receive the result, use the custom_computation tool again with the result in a new calculation

For questions about weather conditions on the moon, use the moon_weather tool
with the appropriate coordinates.

You have access to the following tools: {tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per response, in the following format (without any additional text, preamble, or code blocks):

{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}

Examples:
1. For the custom_computation tool with a simple calculation:
{{"action": "custom_computation", "action_input": "2 + 2"}}

2. For the custom_computation tool with a multi-step calculation (e.g., "Add 5 and 7, then multiply by 2"):
   First step: {{"action": "custom_computation", "action_input": "5 + 7"}}
   After getting result 12, second step: {{"action": "custom_computation", "action_input": "12 * 2"}}

3. For the moon_weather tool:
{{"action": "moon_weather", "action_input": {{"latitude": 40.0, "longitude": 150.0}}}}

4. For general knowledge questions:
{{"action": "Final Answer", "action_input": "The Eiffel Tower is a landmark in Paris, France."}}

Follow this format exactly and do not include any markdown formatting like ```json or ```
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
        # Try with structured parsing first
        result = agent_executor.invoke(
            {"input": query},
            {"max_iterations": max_iterations}
        )
        return result["output"]
    except Exception as e:
        print(f"Error during agent execution: {e}")

        # If there's an error and it seems to be a general knowledge question,
        # try again with a direct approach using our LLM wrapper
        if "calculation" not in query.lower() and not any(word in query.lower() for word in ["moon", "weather", "coordinate"]):
            try:
                print("Attempting direct response for general knowledge question...")
                model_version = "qwen2.5:1.5b"  # Same as in create_agent
                llm = DeepSeekLLM(model_version=model_version)

                # Format a simple prompt for general knowledge
                prompt = f"""You are a helpful assistant answering a general knowledge question. 
Please provide a direct and helpful response to this question:

{query}

Provide your answer in a clear and concise manner."""

                response = llm._call(prompt)
                return response
            except Exception as direct_error:
                print(f"Error with direct approach: {direct_error}")

        # If all else fails, return error message
        return f"The agent encountered an error or exceeded the maximum number of iterations. Error: {e}"
