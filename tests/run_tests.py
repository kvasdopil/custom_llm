#!/usr/bin/env python3
"""
Test runner script for DeepSeek R1 LangGraph Agent Integration Tests
"""
import unittest
import sys
from tests.test_agent import TestDeepSeekAgentIntegration


def run_tests():
    """Run all agent integration tests with detailed output"""
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDeepSeekAgentIntegration)

    # Run the tests with more detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success/failure code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    print("Running DeepSeek R1 LangGraph Agent Integration Tests...")
    sys.exit(run_tests())
