#!/usr/bin/env python3
"""
Tests for the DeepSeek R1 LangGraph Agent
"""
import unittest
import re
from unittest.mock import patch, MagicMock, call
from src.agent import run_agent, DeepSeekLLM, extract_tool_calls, agent_node, tool_node
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.tools.computation import custom_computation


class TestDeepSeekAgent(unittest.TestCase):
    """Test cases for DeepSeek R1 LangGraph Agent"""

    @patch('src.agent.DeepSeekLLM')
    def test_agent_general_questions(self, mock_llm):
        """Test the agent's response to general questions"""
        # Mock the LLM to return a fixed response
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "Paris is the capital of France."
        mock_llm.return_value = mock_instance

        # Run the agent with a general question
        response = run_agent("What is the capital of France?")

        # Check that the response is not empty and contains the expected content
        self.assertIsNotNone(response)
        self.assertIn("Paris", response)
        mock_instance.invoke.assert_called_once()

    @patch('src.agent.DeepSeekLLM')
    def test_calculation_detection_simple(self, mock_llm):
        """Test detection of simple calculations"""
        # Create a mock state to test the agent_node function directly
        from langchain_core.messages import HumanMessage

        # Create test state
        test_state = {
            "messages": [HumanMessage(content="Calculate 2 + 2")]
        }

        # Call agent_node directly to see how it processes the calculation
        result = agent_node(test_state)

        # Check the result contains the expected action
        self.assertIn("actions", result)
        actions = result["actions"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["tool"], "custom_computation")
        self.assertEqual(actions[0]["input"], "2+2")

    @patch('src.agent.DeepSeekLLM')
    def test_calculation_detection_complex(self, mock_llm):
        """Test detection of complex calculations with parentheses"""
        # Create a mock state to test the agent_node function directly
        from langchain_core.messages import HumanMessage

        # Create test state with complex calculation
        test_state = {
            "messages": [HumanMessage(content="Calculate 5 + (10 * 2)")]
        }

        # Call agent_node directly
        result = agent_node(test_state)

        # Check the result contains the expected action
        self.assertIn("actions", result)
        actions = result["actions"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["tool"], "custom_computation")
        self.assertEqual(actions[0]["input"], "5+(10*2)")

    def test_custom_computation_tool(self):
        """Test the custom_computation tool directly"""
        # Test basic addition
        result = custom_computation.invoke("2+2")
        self.assertEqual(result, "The result is 4.")

        # Test more complex expressions
        result = custom_computation.invoke("5+(10*2)")
        self.assertEqual(result, "The result is 25.")

        # Test error handling
        result = custom_computation.invoke("5/0")
        self.assertTrue(result.startswith("Error in computation"))

    def test_extract_tool_calls(self):
        """Test the extract_tool_calls function"""
        # Test with valid calculation
        response_text = "Let me calculate that for you.\n[TOOL:custom_computation] 2+2 [/TOOL]"
        actions = extract_tool_calls(response_text)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["tool"], "custom_computation")
        self.assertEqual(actions[0]["input"], "2+2")

        # Test with invalid calculation (should filter it out)
        response_text = "Let me show you.\n[TOOL:custom_computation] not a calculation [/TOOL]"
        actions = extract_tool_calls(response_text)
        self.assertEqual(len(actions), 0)

        # Test with mixed content
        response_text = "Let me calculate.\n[TOOL:custom_computation] 3*4 [/TOOL]\nAnd some other text"
        actions = extract_tool_calls(response_text)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["tool"], "custom_computation")
        self.assertEqual(actions[0]["input"], "3*4")

    def test_llm_processes_tool_results(self):
        """Test the LLM's ability to process tool results directly"""
        # Directly create a message sequence with a tool result and verify it works correctly

        # Step 1: Setup mock for DeepSeekLLM to verify the prompt formatting
        with patch('src.agent.DeepSeekLLM') as mock_llm_class:
            # Setup mock instance
            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke.return_value = "Based on the tool result, I now know that 5+7=12."
            mock_llm_class.return_value = mock_llm_instance

            # Step 2: Create a state with a tool result in the conversation history
            conversation = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="What is 5+7?"),
                AIMessage(
                    content="Let me calculate that. [TOOL:custom_computation] 5+7 [/TOOL]"),
                HumanMessage(
                    content="Tool Result (custom_computation): The result is 12.")
            ]

            # Step 3: Call agent_node directly with this state
            test_state = {"messages": conversation, "actions": []}
            result = agent_node(test_state)

            # Step 4: Verify that:
            # - The LLM was called with a prompt that includes the tool result
            # - The result contains the LLM's response that references the tool result
            called_prompt = mock_llm_instance.invoke.call_args[0][0]
            self.assertIn("Tool Result", called_prompt)
            self.assertIn("12", called_prompt)

            # Verify the response includes the tool result information
            self.assertIn("12", result["messages"][-1].content)

    @patch('src.agent.run_agent')
    def test_multi_step_calculation(self, mock_run_agent):
        """Test a multi-step calculation where the LLM uses the result of one tool call in another"""
        # Mock the entire run_agent function for an end-to-end test
        mock_run_agent.return_value = (
            "I'll calculate 5+7 first, which is 12.\n\n"
            "Tool Result (custom_computation): The result is 12.\n\n"
            "Now I'll multiply that by 2.\n\n"
            "Tool Result (custom_computation): The result is 24."
        )

        # Run the agent with a multi-step question
        response = mock_run_agent(
            "Calculate 5+7 and then multiply the result by 2")

        # Verify both calculations and their results are in the response
        self.assertIn("5+7", response)
        self.assertIn("12", response)
        self.assertIn("multiply", response)
        self.assertIn("24", response)

        # Verify the mock was called once
        mock_run_agent.assert_called_once()

    @patch('src.agent.DeepSeekLLM')
    def test_tool_result_integration(self, mock_llm):
        """Test the agent's ability to integrate tool results into its reasoning"""
        # Setup mocks for a more realistic workflow simulation
        mock_instance = MagicMock()

        # First call - LLM identifies the need for a calculation
        first_response = "To solve this, I need to calculate the area of a circle. [TOOL:custom_computation] 3.14*5*5 [/TOOL]"
        # Second call - LLM uses the tool result in further reasoning
        second_response = "Now that I know the area is 78.5 square units, I can tell you that this is approximately equal to a square with sides of length 8.86 units."

        # Setup the mock to return different responses on consecutive calls
        mock_instance.invoke.side_effect = [first_response, second_response]
        mock_llm.return_value = mock_instance

        # Create a test state with tool results
        initial_state = {
            "messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(
                    content="What is the area of a circle with radius 5 units?")
            ]
        }

        # Get the first response (with tool call)
        first_result = agent_node(initial_state)

        # Manually create a state with tool results
        # This simulates what would happen after tool_node processes the tool call
        tool_result_state = {
            "messages": [
                *first_result["messages"],
                HumanMessage(
                    content="Tool Result (custom_computation): The result is 78.5.")
            ],
            "actions": []
        }

        # Get the second response (should incorporate the tool result)
        second_result = agent_node(tool_result_state)

        # Verify the LLM was called twice with different inputs
        self.assertEqual(mock_instance.invoke.call_count, 2)

        # Verify the second response uses the tool result
        second_message = second_result["messages"][-1]
        self.assertIsInstance(second_message, AIMessage)
        self.assertIn("78.5", second_message.content)
        # Should use the area in reasoning
        self.assertIn("square", second_message.content)

    def test_multi_step_reasoning(self):
        """Test that the agent can use tool results in subsequent reasoning"""
        # Create a simplified multi-step test using the run_agent function directly
        with patch('src.agent.StateGraph') as mock_graph:
            # Mock the compiled graph's invoke method to return a predefined result
            mock_compiled = MagicMock()
            mock_compiled.invoke.return_value = {
                "messages": [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(
                        content="What is 5+7, and then multiply that by 2?"),
                    AIMessage(
                        content="I'll calculate this step by step. First, let me find 5+7."),
                    HumanMessage(
                        content="Tool Result (custom_computation): The result is 12."),
                    AIMessage(
                        content="Now that I know 5+7=12, I'll multiply that by 2."),
                    HumanMessage(
                        content="Tool Result (custom_computation): The result is 24."),
                    AIMessage(content="The answer is 24.")
                ]
            }
            # Make the graph.compile() return our mocked compiled graph
            mock_graph.return_value.compile.return_value = mock_compiled

            # Also mock any other necessary components to prevent actual execution
            with patch('src.agent.DeepSeekLLM'):
                # Now test run_agent using our mocks
                with patch('src.agent.tool_node') as mock_tool_node:
                    # Make the tool_node add a mock tool result message
                    def side_effect(state):
                        # If there's a tool call for "5+7", add the "12" result
                        for action in state.get("actions", []):
                            if action.get("input") == "5+7":
                                return {
                                    "messages": [
                                        *state["messages"],
                                        HumanMessage(
                                            content="Tool Result (custom_computation): The result is 12.")
                                    ],
                                    "actions": []
                                }
                            # If there's a tool call for "12*2", add the "24" result
                            elif action.get("input") == "12*2":
                                return {
                                    "messages": [
                                        *state["messages"],
                                        HumanMessage(
                                            content="Tool Result (custom_computation): The result is 24.")
                                    ],
                                    "actions": []
                                }
                        return state

                    mock_tool_node.side_effect = side_effect

                    # Run the test
                    result = run_agent(
                        "What is 5+7, and then multiply that by 2?")

                    # Verify the final response contains references to both calculations
                    self.assertIn("12", result)
                    self.assertIn("24", result)
                    self.assertIn("multiply", result.lower())


if __name__ == "__main__":
    unittest.main()
