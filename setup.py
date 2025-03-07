"""
DeepSeek R1 LangGraph Agent setup
"""
from setuptools import setup, find_packages

setup(
    name="deepseek-agent",
    version="0.1.0",
    description="DeepSeek R1 agent using LangGraph",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "langchain-core>=0.1.0",
        "langgraph>=0.0.17",
        "requests>=2.28.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.5.0",
        "langchain>=0.1.0",
    ],
)
