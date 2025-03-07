import { ChatOllama } from "@langchain/ollama"
import { z } from "zod"
import { tool } from "@langchain/core/tools"


const calculationSchema = z.object({
  a: z.number().describe("first number"),
  b: z.number().describe("second number"),
})

const additionTool = tool(
  async ({ a, b }) => {
    return a + b
  },
  {
    name: "addition",
    description: "Add numbers.",
    schema: calculationSchema,
  }
)

const multiplicationTool = tool(
  async ({ a, b }) => {
    return a * b
  },
  {
    name: "multiplication",
    description: "Multiply numbers.",
    schema: calculationSchema,
  }
)

const llm = new ChatOllama({
  model: 'qwen2.5:14b', // 'qwen2.5:1.5b',
  baseUrl: "http://localhost:11434",
  temperature: 0.0, 
});

const llmWithTools = llm.bindTools([
  additionTool,
  multiplicationTool
])

const toolMapping = {
  addition: additionTool,
  multiplication: multiplicationTool
}

let llmOutput = await llmWithTools.invoke("Add 3 and 4, multiply the result by 77")

for (const toolCall of llmOutput.tool_calls) {
  console.log("🛠️ Tool:", toolCall.name, "Args:", toolCall.args)
}

// Invoke the tools
for (const toolCall of llmOutput.tool_calls) {
  const functionToCall = toolMapping[toolCall.name]
  const result = await functionToCall.invoke(toolCall.args)
  console.log("🤖 Result for:", toolCall.name, "with:", toolCall.args, "=", result)
}
