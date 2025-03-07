#!/usr/bin/env python3
"""
Benchmark script for DeepSeek R1 LangGraph Agent
"""
from src.agent import run_agent
import time
import argparse
import json
import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


# Sample questions to test the agent with
DEFAULT_QUESTIONS = [
    "What is artificial intelligence?",
    "Calculate 42 * 13",
    "What is the capital of France?",
    "Calculate 5 + (10 * 2)",
    "Who is the author of 'Pride and Prejudice'?",
    "Calculate 100 / 4",
    "What is machine learning?",
    "Calculate 7 * 8 + 3",
    "What is the meaning of life?",
    "Calculate 2^8"
]


def run_benchmark(questions=None, output_file=None, max_iterations=5):
    """
    Run benchmark tests on the agent with a set of questions.

    Args:
        questions: List of questions to test with
        output_file: File to save results to (JSON format)
        max_iterations: Maximum number of iterations for each agent run

    Returns:
        Dictionary with benchmark results
    """
    if questions is None:
        questions = DEFAULT_QUESTIONS

    results = []
    total_time = 0

    print(f"Running benchmark with {len(questions)} questions...")

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Testing: {question}")
        start_time = time.time()

        response = run_agent(question, max_iterations=max_iterations)

        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

        # Check if it's a calculation (simple heuristic)
        is_calculation = "calculate" in question.lower() or any(
            symbol in question for symbol in "+-*/^"
        )

        # Check if the response contains a numerical result
        has_computation_result = "The result is" in response

        # This is a simple success heuristic - updated for the new agent implementation
        success = (
            has_computation_result if is_calculation else (
                len(response.strip()) > 20)
        )

        result = {
            "question": question,
            "is_calculation": is_calculation,
            "time_seconds": round(elapsed_time, 2),
            "response_length": len(response),
            "used_tool": has_computation_result,
            "success": success,
            "response": response[:200] + "..." if len(response) > 200 else response
        }

        results.append(result)

        print(
            f"Time: {result['time_seconds']:.2f}s, Success: {result['success']}")

    summary = {
        "total_questions": len(questions),
        "total_time": round(total_time, 2),
        "average_time": round(total_time / len(questions), 2),
        "success_rate": sum(r["success"] for r in results) / len(results),
        "results": results
    }

    print(f"\nBenchmark complete!")
    print(f"Total time: {summary['total_time']:.2f}s")
    print(f"Average time per question: {summary['average_time']:.2f}s")
    print(f"Success rate: {summary['success_rate'] * 100:.1f}%")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {output_file}")

    return summary


def main():
    """Main entry point for the benchmark script"""
    parser = argparse.ArgumentParser(
        description='Benchmark the DeepSeek R1 LangGraph Agent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--questions-file',
        type=str,
        help='Path to a file containing questions, one per line'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        default='benchmark_results.json',
        help='Path to save the benchmark results'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='Maximum number of iterations for each agent run'
    )

    args = parser.parse_args()

    questions = DEFAULT_QUESTIONS
    if args.questions_file:
        try:
            with open(args.questions_file, 'r') as f:
                questions = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading questions file: {e}")
            return 1

    run_benchmark(
        questions=questions,
        output_file=args.output_file,
        max_iterations=args.max_iterations
    )

    return 0


if __name__ == "__main__":
    main()
