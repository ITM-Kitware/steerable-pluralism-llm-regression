import os
import json
import pandas as pd
import numpy as np

from data_utils import stratified_sampling

MAP_AGREE = {'0':-1.0, '1':0.0, '2':1.0, 'nan':0.0}
MORALS = ['care','fairness','liberty','loyalty','authority','sanctity']

def get_multi_answer_subset(df):
    # Get subset with multiple answer options
    result = []
    # Iterate through each group of 'Q' where there are at least two distinct 'A' values
    for q_value, group in df.groupby('Q'):
        # Check if the group has more than one unique 'A' value
        if group['A'].nunique() > 1:
            # make sure no nans
            nans = False
            for index, row in group.iterrows():
                if pd.isna(row['A_agrees']):
                    nans = True
            if not nans:
                result.append(group)
    # Concatenate the result list into a DataFrame
    result_df = pd.concat(result)
    return result_df 

def get_label(a_df):
    annotations = []
    for index, row in a_df.iterrows():
        agreement = MAP_AGREE[row['A_agrees']]
        annotations.append(agreement*np.array(eval(row['moral-vector'])))
    annotations = annotations[:3] # most have just 3, only use first 3 if there are more
    annotations = np.array(annotations).sum(axis=0)
    normalized_annotations = (annotations +3)/6 # min=-3, max=3
    labels = {}
    for i in range(len(MORALS)):
        labels[MORALS[i]] = round(normalized_annotations[i],2)
    return labels

def get_rot_reasoning(a_df):
    rot_reasoning = {}
    for i in range(len(MORALS)):
        rot_reasoning[MORALS[i]] = ''
        count = 0
        for index, row in a_df.iterrows():
            if count < 3: # first 3 only
                if eval(row['moral-vector'])[i] == 1:
                    if row['A_agrees'] == '0':
                        rot_reasoning[MORALS[i]] += f"The response disagrees with the {MORALS[i]} rule of thumb: {row['rot']} "
                    elif row['A_agrees'] == '2':
                        rot_reasoning[MORALS[i]] += f"The response agrees with the {MORALS[i]} rule of thumb: {row['rot']} "
            count += 1
        if rot_reasoning[MORALS[i]] == '':
            rot_reasoning[MORALS[i]] = f"The response is neutral with respect to {MORALS[i]}."
    return rot_reasoning

def reformat_data(result_df):
    # Iterate through unique questions
    dataset = []
    for question in result_df['Q'].unique():
        instance = {'prompt':question, 'responses':[], 'labels':[], 'reasonings':[]}
        q_subset = result_df[result_df['Q'] == question]
        for answer in q_subset['A'].unique():
            a_subset = q_subset[q_subset['A'] == answer]
            instance['responses'].append(answer)
            instance['labels'].append(get_label(a_subset))
            instance['reasonings'].append(get_rot_reasoning(a_subset))
        dataset.append(instance)
    return dataset

if __name__ == '__main__':

    df = pd.read_csv('mic/MIC.csv')
    print(f"Unique questions: {df['Q'].nunique()}")
    result_df = get_multi_answer_subset(df)
    print(f"Unique questions with multiple answers: {result_df['Q'].nunique()}")
    dataset = reformat_data(result_df)

    data_dir = 'mic/'
    target_dir = '../alignment_targets/mic/'
    num_label_levels = 7 # -3 through 3

    # Split
    train_data, eval_data = stratified_sampling(dataset, len(MORALS), num_label_levels)
    print(f'Train data: {len(train_data)}')
    print(f'Eval data: {len(eval_data)}')
    with open(os.path.join(data_dir, "train.json"), "w") as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(data_dir, "eval.json"), "w") as f:
        json.dump(eval_data, f, indent=4)
    print(f'Saved to {data_dir}')