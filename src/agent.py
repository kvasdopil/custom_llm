#!/usr/bin/env python3
"""
DeepSeek R1 LangGraph Agent Implementation
"""
from typing import ClassVar, Dict, List, Optional, TypedDict, Union, Any
import re

# LangGraph imports
from langgraph.graph import END, StateGraph

# LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.llms.base import LLM
from pydantic import BaseModel, Field

# For the model connection
import requests
import json

# Custom tool import
from src.tools.computation import custom_computation


class DeepSeekLLM(LLM):
    """Wrapper for DeepSeek model."""
    model_name: ClassVar[str] = "deepseek-r1"
    model_version: str = "deepseek-r1:1.5b"

    def __init__(self, model_version: str = "deepseek-r1:1.5b"):
        super().__init__()
        self.model_version = model_version

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


# Helper function to format messages for the model
def format_messages_to_prompt(messages: List[Any]) -> str:
    """Format a list of messages into a single prompt string."""
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"AI: {msg.content}")
        elif isinstance(msg, SystemMessage):
            formatted.append(f"System: {msg.content}")
        else:
            formatted.append(f"Other: {str(msg)}")
    return "\n".join(formatted)


# Check for tool usage patterns in the LLM response
def extract_tool_calls(response_text: str):
    """Extract tool calls from LLM response text."""
    tool_pattern = r'\[TOOL:([^\]]+)\](.*?)\[/TOOL\]'
    tool_matches = re.findall(tool_pattern, response_text, re.DOTALL)

    actions = []
    for tool_name, tool_input in tool_matches:
        # Only add valid tool calls - check if the input is a valid math expression
        tool_name = tool_name.strip()
        tool_input = tool_input.strip()

        if tool_name == "custom_computation":
            # Try to validate the math expression
            try:
                # Simple validation - expression should contain only valid math characters
                if re.match(r'^[\d\s\+\-\*\/\(\)\.\%\^\,]+$', tool_input):
                    actions.append({
                        "tool": tool_name,
                        "input": tool_input
                    })
            except:
                # If validation fails, don't add the action
                pass
        else:
            # For other tools, add them without validation
            actions.append({
                "tool": tool_name,
                "input": tool_input
            })

    return actions


# Agent node that processes the current state
def agent_node(state: AgentState) -> Dict:
    """Agent node that generates responses based on the current state."""
    messages = state["messages"]

    # Create the system message if not present
    has_system = any(isinstance(m, SystemMessage) for m in messages)
    if not has_system:
        messages = [
            SystemMessage(content=(
                "You are a helpful AI assistant that can use tools to assist users. "
                "For any mathematical calculation, you MUST use the custom_computation tool. "
                "DO NOT calculate the result yourself. "
                "Use the tool by writing [TOOL:custom_computation] followed by the expression, "
                "then [/TOOL]. For example: [TOOL:custom_computation] 2+2 [/TOOL]"
            )),
            *messages
        ]

    # Check if the latest message contains a calculation request
    if messages and isinstance(messages[-1], HumanMessage):
        content = messages[-1].content.lower()
        if any(word in content for word in ["calculate", "computation", "multiply", "divide", "add", "subtract", "*", "/", "+", "-"]):
            # Try to extract a full mathematical expression
            # Look for expressions with parentheses first
            full_expr = re.search(
                r'(\d+\s*[\+\-\*\/]\s*[\(\)\d\s\+\-\*\/\.]+)', content)
            if full_expr:
                expression = re.sub(r'\s+', '', full_expr.group(1))
            else:
                # Then try to find simpler expressions
                simple_expr = re.search(r'(\d+\s*[\+\-\*\/]\s*\d+)', content)
                if simple_expr:
                    expression = re.sub(r'\s+', '', simple_expr.group(1))
                else:
                    # No expression found, let the LLM handle it
                    expression = None

            if expression:
                ai_message = AIMessage(content=(
                    f"I'll calculate {expression} for you using the custom_computation tool.\n"
                    f"[TOOL:custom_computation] {expression} [/TOOL]"
                ))
                return {"messages": [*messages, ai_message], "actions": [{
                    "tool": "custom_computation",
                    "input": expression
                }]}

    # Format messages for the LLM
    prompt = format_messages_to_prompt(messages)

    # Call the LLM
    llm = DeepSeekLLM()
    response_text = llm.invoke(prompt)

    # Create the AI message
    ai_message = AIMessage(content=response_text)

    # Extract valid tool calls
    actions = extract_tool_calls(response_text)

    return {"messages": [*messages, ai_message], "actions": actions}


# Tool execution node
def tool_node(state: AgentState) -> Dict:
    """Tool node that executes tools based on the actions in the state."""
    messages = state["messages"]
    actions = state["actions"]

    # If no actions, just return the state unchanged with empty actions
    if not actions:
        return {"messages": messages, "actions": []}

    results = []
    for action in actions:
        tool_name = action["tool"]
        tool_input = action["input"]

        # Execute the appropriate tool
        if tool_name == "custom_computation":
            try:
                result = custom_computation(tool_input)
                results.append(f"Tool Result ({tool_name}): {result}")
            except Exception as e:
                results.append(f"Tool Error ({tool_name}): {str(e)}")
        else:
            results.append(f"Unknown tool: {tool_name}")

    # Return the updated state with empty actions to ensure we stop
    if results:
        tool_results_message = HumanMessage(content="\n".join(results))
        return {"messages": [*messages, tool_results_message], "actions": []}

    # Always clear the actions to avoid recursion
    return {"messages": messages, "actions": []}


def create_agent():
    """Create a simple LangGraph agent workflow."""
    # Create the state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Add the entrypoint
    workflow.set_entry_point("agent")

    # Add an edge from agent to tools
    workflow.add_edge("agent", "tools")

    # Add conditional edges for tools node
    workflow.add_conditional_edges(
        "tools",
        lambda state: {
            # Check if there are more actions to process
            # If no actions, end the workflow
            "agent": len(state.get("actions", [])) > 0,
            END: len(state.get("actions", [])) == 0
        }
    )

    # Compile the graph
    return workflow.compile()


def run_agent(query: str, max_iterations: int = 5):
    """Run the agent with a query.

    Args:
        query: The user's query to process
        max_iterations: Maximum number of iterations to prevent infinite loops

    Returns:
        The agent's response
    """
    agent = create_agent()

    # Initialize the state
    initial_state = AgentState(
        messages=[HumanMessage(content=query)],
        actions=[]
    )

    # Run the agent with a timeout mechanism
    try:
        result = agent.invoke(
            initial_state, {"recursion_limit": max_iterations})
    except Exception as e:
        print(f"Error during agent execution: {e}")
        return f"The agent encountered an error or exceeded the maximum number of iterations. Error: {e}"

    # Build a complete response including the final answer and any tool results
    final_response = []
    tool_results = []

    # Process all messages
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            # Clean up the response by removing tool annotations
            content = re.sub(r'\[TOOL:.*?\].*?\[/TOOL\]',
                             '', message.content, flags=re.DOTALL)
            if content.strip():
                final_response.append(content.strip())
        elif isinstance(message, HumanMessage) and "Tool Result" in message.content:
            tool_results.append(message.content)

    # Combine final response with tool results
    if tool_results:
        final_response.append("\n".join(tool_results))

    # Return the final combined response
    if final_response:
        return "\n\n".join(final_response)

    return "No response generated."


if __name__ == "__main__":
    # This allows running directly from this file for testing
    test_query = "Calculate 23 * 17"
    print(f"Test query: {test_query}")
    test_response = run_agent(test_query)
    print("\nTest response:")
    print(test_response)
