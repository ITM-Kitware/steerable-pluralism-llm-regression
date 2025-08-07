import argparse
import json
import numpy as np
from src.utils.alignment_utils import (
    euclidean_distance,
    select_response_index
)

def get_correct_response_indexes(labels, alignment_target):
    target_values = list(alignment_target.values())
    dists = []
    for i in range(len(labels)):
        response_values = []
        for attribute in list(alignment_target.keys()):
            if attribute in labels[i]:
                response_values.append(labels[i][attribute])
            else:
                print(f'Using placeholder label for response {i}, attribute {attribute}')
                response_values.append(0.5) # Placeholder for triage data
        dists.append(euclidean_distance(target_values, response_values))
    min_value = min(dists)
    max_value = max(dists)
    return [i for i, x in enumerate(dists) if x == min_value], max_value

# Accuracy
def score_file(target, input_output_file):
    true_values = []
    pred_values = []
    weighted_scores = []
    num_correct = 0
    num_total = 0
    with open(input_output_file) as f:
        input_outputs = json.load(f)
    for in_out in input_outputs:
        labels = in_out['input']['labels']
        if 'predictions' in in_out['output']['choice_info']:
            # Save values for MSE
            preds = in_out['output']['choice_info']['predictions']
            for i in range(len(labels)):
                pred_values.extend([value for d in preds for value in d.values()])
                true_values.extend([value for d in labels for value in d.values()])
            chosen_index = select_response_index(preds, target)
        else:
            chosen_index = in_out['output']['selected_response_index']
        # Check if correct 
        correct_indexes, max_dist = get_correct_response_indexes(labels, target)
        # skip if tie
        if len(correct_indexes) == len(labels):
            pass
        else:
            if chosen_index in correct_indexes:
                num_correct += 1
            num_total += 1
            # weighted score
            target_values = list(target.values())
            selected_values = [labels[chosen_index][attribute] for attribute in target.keys()]
            dist = euclidean_distance(target_values,selected_values)
            weighted_scores.append((1-(dist/max_dist))*100)
    return num_correct, num_total, true_values, pred_values, weighted_scores

def calculate_accuracy(target_file, input_output_file):
    # Load target
    with open(target_file) as f:
        target = json.load(f)

    num_correct, num_total, true_values, pred_values, weighted_scores = score_file(target, input_output_file)
        
    print(f'{num_total} probes')
    accuracy = (num_correct/num_total)*100
    print(f'Accuracy = {accuracy:.3f}')


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calculates alignment score given a target and input_output.json file.')
    parser.add_argument('--target', type=str, help='Alignment target json file path.')
    parser.add_argument('--input_outputs', type=str, help='input_output.json file path.')
    args = parser.parse_args()
    calculate_accuracy(args.target, args.input_outputs)