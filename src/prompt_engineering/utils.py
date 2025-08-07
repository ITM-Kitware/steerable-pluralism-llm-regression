import itertools

def list_to_multiple_choice(responses):
    # Create a dictionary with letter keys (A, B, C, ...) and corresponding responses
    choices = {chr(65 + i): string for i, string in enumerate(responses)}
    return choices

# Function borrowed from
# https://docs.python.org/3/library/itertools.html#itertools.batched
# (since itertools.batched is only available in Python 3.12 or newer):
def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch

def run_in_batches(inference_function, inputs, batch_size):
    ''' Batch inference to avoid out of memory error'''
    outputs = []
    for batch in batched(inputs, batch_size):
        output = inference_function(list(batch))
        if not isinstance(output, list):
            output = [output]
        outputs.extend(output)
    return outputs
         