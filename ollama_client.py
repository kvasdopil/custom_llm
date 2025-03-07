from ollama import ChatResponse, chat
from tools import (
    shmabigate_two_numbers,
    shmabigate_two_numbers_tool,
    combombulate_two_numbers,
    combombulate_two_numbers_tool,
)

messages = [
    {'role': 'user', 'content': 'Calculate the result of (3 $ 4) $ 5'}]
print('Prompt:', messages[0]['content'])

available_functions = {
    'shmabigate_two_numbers': shmabigate_two_numbers,
    'combombulate_two_numbers': combombulate_two_numbers,
}

tools = [shmabigate_two_numbers_tool, combombulate_two_numbers_tool]

model = 'llama3.1'
response: ChatResponse = chat(
    model,
    messages=messages,
    tools=tools,
)

if response.message.tool_calls:
    # There may be multiple tool calls in the response
    for tool in response.message.tool_calls:
        # Ensure the function is available, and then call it
        if function_to_call := available_functions.get(tool.function.name):
            print('Calling function:', tool.function.name)
            print('Arguments:', tool.function.arguments)
            output = function_to_call(**tool.function.arguments)
            print('Function output:', output)
        else:
            print('Function', tool.function.name, 'not found')

    # Add the function response to messages for the model to use
    messages.append(response.message)
    messages.append({'role': 'tool', 'content': str(
        output), 'name': tool.function.name})

    # Get final response from model with function outputs
    final_response = chat(model, messages=messages, tools=tools)
    print('Final response:', final_response.message)

else:
    print('No tool calls returned from model')
