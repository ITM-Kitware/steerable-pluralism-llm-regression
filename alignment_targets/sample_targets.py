import argparse
from itertools import product
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import json

random.seed(42)

def get_all_possible_targets(dataset='mic'):
    if dataset == 'mic':
        attributes = ['care','fairness','liberty','loyalty','authority','sanctity']
        num_label_levels = 7
    elif dataset == 'helpsteer2':
        attributes = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
        num_label_levels = 5
   
    values = [round(i/(num_label_levels-1), 2) for i in range(num_label_levels)]
    combinations = list(product(values, repeat=len(attributes)))
    targets = [dict(zip(attributes, combo)) for combo in combinations]
    # remove neutral 0.5 values from targets
    for i in range(len(targets)):
        targets[i] = {key: value for key, value in targets[i].items() if value != 0.5}
    
    return targets

def plot_and_save_targets(dataset, targets, target_dir):
    
    os.makedirs(target_dir, exist_ok=True)
    
    # Save targets
    for target_index in range(len(targets)):
        target = targets[target_index]
        with open(f'{target_dir}target{target_index:02d}.json', 'w') as f:
            json.dump(target, f)

    # plot targets
    if dataset == 'mic':
        attributes = ['care','fairness','liberty','loyalty','authority','sanctity']
        num_label_levels = 7
    elif dataset == 'helpsteer2':
        attributes = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
        num_label_levels = 5

    if len(targets[0]) > 2: 
        # Create a polar plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        num_angles = len(targets[0])
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        angles = np.append(angles, angles[0])

        # Customize the compass directions
        ax.set_xticks(np.linspace(0, 2 * np.pi, num_angles, endpoint=False))

        # Customize the radial axis
        ax.set_rmax(1.0)  # Set the maximum radius
        ticks = [round(i/(num_label_levels-1), 2) for i in range(num_label_levels)]
        ax.set_rticks(ticks)  # Set the radial tick positions

        cap_atts = [att.capitalize() for att in attributes]
        ax.set_xticklabels(cap_atts)
        ax.tick_params(axis='x', pad=20) 

        # Plot targets
        for i in range(len(targets)):
            values1 = np.array(targets[i].values())
            values1 = np.append(values1, values1[0])
            ax.plot(angles, values1, marker='o', label=f'Target {i+1}') 
            ax.fill(angles, values1, alpha=0.2) 
        plt.savefig('{target_dir}target{i:02d}_plot.png')


def sample(dataset, num_targets, num_attributes):
    targets = get_all_possible_targets(dataset)
    n_targets = [target for target in targets if len(target.keys()) == int(num_attributes)]
    sampled_targets = random.sample(n_targets, num_targets)
    
    # save sampled targets
    target_dir = f'sampled_{dataset}/'
    os.makedirs(target_dir, exist_ok=True)
    for target_index in range(num_targets):
        target = sampled_targets[target_index]
        with open(f'{target_dir}{num_attributes}attributes_target{target_index:02d}.json', 'w') as f:
            json.dump(target, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Uniformly samples targets.')
    parser.add_argument('--dataset', type=str, help='mic or helpsteer2', required=True)
    parser.add_argument('--num_targets', type=int, help='Number of targets', required=True)
    parser.add_argument('--num_attributes', type=int, help='Number of attributes in each target', required=True)
    args = parser.parse_args()
    sample(args.dataset, args.num_targets, args.num_attributes)