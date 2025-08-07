import os
import json
from datasets import load_dataset

from data_utils import stratified_sampling

ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

def normalize_labels(labels):
    min_val = 0
    max_val = 4
    normalized_labels = {
        key: [(value - min_val) / (max_val - min_val) for value in values]
        for key, values in labels.items()
    }
    return normalized_labels

def reformat(data):
    reformatted_data = []
    for i in range(0, len(data)-2, 2):
        example = data[i:i+2]
        reformatted_example = {}
        reformatted_example['prompt'] = example['prompt'][0]
        reformatted_example['responses'] = example['response']
        labels = {}
        for attribute in ['helpfulness', 'correctness', 'coherence','complexity','verbosity']:
            labels[attribute] = example[attribute]
        # Normalize so 0-4 becomes 0-1
        labels = normalize_labels(labels)
        # Convert to a list of dictionaries
        labels_list_of_dicts = [dict(zip(labels.keys(), values)) for values in zip(*labels.values())]
        reformatted_example['labels'] = labels_list_of_dicts
        reformatted_data.append(reformatted_example)
    return reformatted_data

if __name__ == '__main__':
    out_dir = 'helpsteer2'
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("nvidia/HelpSteer2")

    train = ds['train'] 
    train = reformat(train)
    val = ds['validation']
    val = reformat(val)
    dataset = train+val

    data_dir = 'helpsteer2/'
    target_dir = '../alignment_targets/helpsteer2/'
    num_label_levels = 5 # 0-4

    # Split
    train_data, eval_data = stratified_sampling(dataset, len(ATTRIBUTES), num_label_levels)
    print(f'Train data: {len(train_data)}')
    print(f'Eval data: {len(eval_data)}')
    with open(os.path.join(data_dir, "train.json"), "w") as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(data_dir, "eval.json"), "w") as f:
        json.dump(eval_data, f, indent=4)
    print(f'Saved to {data_dir}')
