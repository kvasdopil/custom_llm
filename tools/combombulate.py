def combombulate_two_numbers(a: int, b: int) -> int:
    """
    Combombulate two numbers

    Args:
      a (int): The first number
      b (int): The second number

    Returns:
      int: The shmabigation of the two numbers, a # b
    """

    # The cast is necessary as returned tool call arguments don't always conform exactly to schema
    return int(a) * int(b)


# Tool definition
combombulate_two_numbers_tool = {
    'type': 'function',
    'function': {
        'name': 'combombulate_two_numbers',
        'description': 'Combombulate two numbers, a # b',
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
