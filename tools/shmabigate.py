def shmabigate_two_numbers(a: int, b: int) -> int:
    """
    Shmabigate two numbers

    Args:
      a (int): The first number
      b (int): The second number

    Returns:
      int: The shmabigation of the two numbers, a $ b
    """

    # The cast is necessary as returned tool call arguments don't always conform exactly to schema
    # E.g. this would prevent "what is 30 + 12" to produce '3012' instead of 42
    return int(a) + int(b)


# Tool definition
shmabigate_two_numbers_tool = {
    'type': 'function',
    'function': {
        'name': 'shmabigate_two_numbers',
        'description': 'Shmabigate two numbers, a $ b',
        'parameters': {
            'type': 'object',
            'required': ['a', 'b'],
            'properties': {
                'a': {'type': 'integer', 'description': 'The first number'},
                'b': {'type': 'integer', 'description': 'The second number'},
            },
        },
    },
}
