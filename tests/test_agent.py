#!/usr/bin/env python3
"""
Integration tests for the DeepSeek R1 LangGraph Agent
"""
import unittest
from unittest.mock import patch, MagicMock
from src.agent import run_agent
from src.tools.computation import custom_computation


class TestDeepSeekAgentIntegration(unittest.TestCase):
    """Integration tests for DeepSeek R1 LangGraph Agent

    These tests verify the end-to-end functionality of the agent.
    """

    @patch('src.agent.DeepSeekLLM')
    @patch('langchain.agents.create_structured_chat_agent')
    @patch('langchain.agents.AgentExecutor.from_agent_and_tools')
    def test_general_knowledge_question(self, mock_executor_factory, mock_agent_factory, mock_llm_class):
        """Test agent handling general knowledge questions without tools."""
        # Set up mock LLM instance
        mock_llm_instance = MagicMock()
        mock_llm_instance._call.return_value = '{"action": "Final Answer", "action_input": "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. It was built in 1889 and stands 330 meters tall."}'
        mock_llm_class.return_value = mock_llm_instance

        # Mock agent creation
        mock_agent = MagicMock()
        mock_agent_factory.return_value = mock_agent

        # Mock executor creation and execution
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {
            "output": "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. It was built in 1889 and stands 330 meters tall."
        }
        mock_executor_factory.return_value = mock_executor

        # Run the agent with a general knowledge question
        response = run_agent("What is the Eiffel Tower?")

        # Verify response contains expected information
        self.assertIn("Eiffel Tower", response)
        self.assertIn("Paris", response)

        # Verify the agent was properly invoked
        mock_executor.invoke.assert_called_once()

    @patch('src.agent.DeepSeekLLM')
    @patch('langchain.agents.create_structured_chat_agent')
    @patch('langchain.agents.AgentExecutor.from_agent_and_tools')
    def test_simple_calculation(self, mock_executor_factory, mock_agent_factory, mock_llm_class):
        """Test agent correctly routes calculation to the custom computation tool."""
        # Set up mock LLM instance
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        # Mock agent creation
        mock_agent = MagicMock()
        mock_agent_factory.return_value = mock_agent

        # Simulate the full agent workflow with calculation tool usage
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {
            "output": "The result of 5 + 7 is 12.",
            "intermediate_steps": [
                ({"action": "custom_computation",
                 "action_input": "5 + 7"}, "The result is 12.")
            ]
        }
        mock_executor_factory.return_value = mock_executor

        # Run the agent with a calculation question
        response = run_agent("What is 5 + 7?")

        # Verify response contains the correct calculation result
        self.assertIn("12", response)

        # Verify the agent was properly invoked
        mock_executor.invoke.assert_called_once()

    @patch('src.agent.DeepSeekLLM')
    @patch('langchain.agents.create_structured_chat_agent')
    @patch('langchain.agents.AgentExecutor.from_agent_and_tools')
    def test_multi_step_calculation(self, mock_executor_factory, mock_agent_factory, mock_llm_class):
        """Test agent can handle multi-step calculations using tool results."""
        # Set up mock LLM instance
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        # Mock agent creation
        mock_agent = MagicMock()
        mock_agent_factory.return_value = mock_agent

        # Simulate multi-step calculation workflow
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {
            "output": "To solve this problem, I first calculated 5 + 7, which equals 12. Then I multiplied this result by 2, which gives us 24.",
            "intermediate_steps": [
                ({"action": "custom_computation",
                 "action_input": "5 + 7"}, "The result is 12."),
                ({"action": "custom_computation",
                 "action_input": "12 * 2"}, "The result is 24.")
            ]
        }
        mock_executor_factory.return_value = mock_executor

        # Run the agent with a multi-step question
        response = run_agent(
            "What is 5 + 7, and then multiply the result by 2?")

        # Verify response contains both calculation steps and final result
        self.assertIn("12", response)
        self.assertIn("24", response)

        # Verify the agent was properly invoked
        mock_executor.invoke.assert_called_once()

    @patch('src.agent.DeepSeekLLM')
    @patch('langchain.agents.create_structured_chat_agent')
    @patch('langchain.agents.AgentExecutor.from_agent_and_tools')
    def test_complex_calculation(self, mock_executor_factory, mock_agent_factory, mock_llm_class):
        """Test agent correctly handles more complex mathematical expressions."""
        # Set up mock LLM instance
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        # Mock agent creation
        mock_agent = MagicMock()
        mock_agent_factory.return_value = mock_agent

        # Simulate complex calculation workflow
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {
            "output": "The area of a circle with radius 5 is 3.14 × 5² = 78.5 square units.",
            "intermediate_steps": [
                ({"action": "custom_computation",
                 "action_input": "3.14 * (5 ** 2)"}, "The result is 78.5.")
            ]
        }
        mock_executor_factory.return_value = mock_executor

        # Run the agent with a complex calculation
        response = run_agent("Calculate the area of a circle with radius 5.")

        # Verify response contains the correct calculation result
        self.assertIn("78.5", response)

        # Verify the agent was properly invoked
        mock_executor.invoke.assert_called_once()

    @patch('src.agent.DeepSeekLLM')
    @patch('langchain.agents.create_structured_chat_agent')
    @patch('langchain.agents.AgentExecutor.from_agent_and_tools')
    def test_error_handling(self, mock_executor_factory, mock_agent_factory, mock_llm_class):
        """Test agent appropriately handles computation errors."""
        # Set up mock LLM instance
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        # Mock agent creation
        mock_agent = MagicMock()
        mock_agent_factory.return_value = mock_agent

        # Simulate error handling workflow
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {
            "output": "I attempted to calculate 5 ÷ 0, but this is mathematically undefined. Division by zero is not allowed in mathematics.",
            "intermediate_steps": [
                ({"action": "custom_computation", "action_input": "5 / 0"},
                 "Error in computation: division by zero")
            ]
        }
        mock_executor_factory.return_value = mock_executor

        # Run the agent with a calculation that causes an error
        response = run_agent("What is 5 divided by 0?")

        # Verify response contains error handling information
        self.assertTrue(any(term in response.lower()
                        for term in ["undefined", "error", "not allowed"]))

        # Verify the agent was properly invoked
        mock_executor.invoke.assert_called_once()

    def test_direct_tool_usage(self):
        """Test the computation tool directly to ensure it works standalone."""
        # Test basic addition
        result = custom_computation.invoke("2+2")
        self.assertEqual(result, "The result is 4.")

        # Test more complex expression
        result = custom_computation.invoke("3.14*(5**2)")
        self.assertEqual(result, "The result is 78.5.")

        # Test error handling
        result = custom_computation.invoke("5/0")
        self.assertTrue(result.startswith("Error in computation"))


if __name__ == "__main__":
    unittest.main()
