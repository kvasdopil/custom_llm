from typing import ClassVar
from langchain.llms.base import LLM
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType, ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel
import requests
import json


class DeepSeekLLM(LLM):
    model_name: ClassVar[str] = "deepseek-r1"

    def _call(self, prompt: str, stop: list = None) -> str:
        print("calling", prompt)
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "deepseek-r1:1.5b",
                "messages": [{"role": "user", "content": prompt}],
            },
            stream=True
        )
        print("streaming response")
        response_text = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    response_json = json.loads(decoded_line)
                    response_text += response_json["message"]["content"]
                    # Print each part of the message as it arrives
                    print(response_json["message"]["content"])
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

        return response_text

    @property
    def _llm_type(self) -> str:
        return "custom_deepseek"

# Define a custom tool using the DeepSeekLLM


class DeepSeekTool(BaseModel):
    llm: DeepSeekLLM = DeepSeekLLM()

    def run(self, input_text: str) -> str:
        # Perform custom computation here
        print("Performing custom computation")
        result = f"Custom computation result for: {input_text}"
        return result

# Initialize the ReAct agent


def initialize_react_agent():
    deepseek_tool = DeepSeekTool()
    tools = [
        Tool(
            name="DeepSeekTool",
            func=deepseek_tool.run,
            description="A tool that performs custom computation for the calling LLM agent."
        )
    ]

    # Define a custom prompt template that explicitly shows the expected format
    prefix = """Answer the following questions as best you can. You have access to the following tools:"""
    suffix = """Begin! Remember to answer in the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought:"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"]
    )

    memory = ConversationBufferMemory(memory_key="chat_history")
    llm_chain = LLMChain(llm=deepseek_tool.llm, prompt=prompt)

    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory
    )

    return agent_executor


# Example usage
if __name__ == "__main__":
    print("Querying the model", flush=True)
    agent = initialize_react_agent()
    response = agent.run(
        "Explain the significance of reinforcement learning in AI.")
    print("Response:", response)
