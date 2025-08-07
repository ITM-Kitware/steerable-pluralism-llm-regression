import os
import json
import time
from transformers import pipeline

# Converts atribute/value pairs to words
def value_to_words(attribute, value):
    attribute_verbage = {
        'helpfulness':{'high':'helpful', 'low':'unhelpful'},
        'correctness':{'high':'correct', 'low':'incorrect'},
        'coherence':{'high':'coherent', 'low':'incoherent'},
        'complexity':{'high':'complex', 'low':'simple'},
        'verbosity':{'high':'verbose','low':'brief'}
    }
    if value == 0.0:
        words = f'very {attribute_verbage[attribute]['low']}'
    elif value == 0.25:
        words = f'{attribute_verbage[attribute]['low']}'
    elif value == 0.5:
        words = f'somewhat {attribute_verbage[attribute]['high']}'
    elif value == 0.5:
        words = f'{attribute_verbage[attribute]['high']}'
    else:
        words = f'very {attribute_verbage[attribute]['high']}'
    return words

# Generates reasoning for a question/response pair
def generate_reasonings(generator, question, response, label):
    gens ={}
    for attribute, value in label.items():
        prompt = f'Question:\n {question}\n Response:\n {response}\n '
        reasoning_start = f'The response is {value_to_words(attribute, value)} because'
        prompt += reasoning_start
        output = generator(prompt, max_length=len(prompt)+20, num_return_sequences=1)
        full_text = output[0]['generated_text']
        completion = full_text[len(prompt):].strip()
        completion = completion.split(".")[0] # Take first sentence only
        reason = f'{reasoning_start} {completion}.'
        print(reason)
        gens[attribute] = reason
    return gens 

# Adds reasonings to data json
def add_reasoning(data, generator):
    data_with_reasonong = []
    for scenario in data:
        question = scenario['prompt']
        reasonings = []
        for i in range(len(scenario['labels'])):
            response = scenario['responses'][i]
            label = scenario['labels'][i]
            reasonings.append(generate_reasonings(generator, question, response, label))
        scenario['reasonings'] = reasonings
        data_with_reasonong.append(scenario)
    return data_with_reasonong

if __name__ == '__main__':
    start_time = time.time()

    generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3", device=3)
    data_dir = 'helpsteer2'
    
    train_json = os.path.join(data_dir, "train.json")
    with open(os.path.join(data_dir, "train.json")) as f:
        train_data = json.load(f)

    train_data_with_reasoning = add_reasoning(train_data, generator)

    with open(os.path.join(data_dir, "train.json"), "w") as f:
        json.dump(train_data, f, indent=4)
    
    print(f'Runtime = {time.time()-start_time} seconds.')
