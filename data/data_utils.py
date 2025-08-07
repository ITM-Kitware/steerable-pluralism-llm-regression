import numpy as np


def stratified_sampling(dataset, num_attributes, num_label_levels):
    # Get value data
    a_to_q_indices = {} # keep track of answer to question indices
    data = []
    a = 0
    for q in range(len(dataset)):
        for choice in dataset[q]['labels']:
            data.append([value for value in choice.values()])
            a_to_q_indices[a] = q
            a += 1
    data = np.array(data)

    # get counts of each attribute/label instance
    counts = []
    for att in range(num_attributes):
        for i in range(num_label_levels):
            value = round(i/(num_label_levels-1), 2)
            counts.append(np.sum(data[:,att]==value))
    counts = np.array(counts)
    
    num_of_each = min(np.array(counts).min(), 20)
    print(f"Num of each attribute/value combo in eval: {num_of_each}")

    train_indices = []
    eval_indices = []
    seen_indices = set([])
    sorted_indices = np.argsort(counts)
    for idx in sorted_indices:
        att_idx = idx//num_label_levels
        val_idx = idx%num_label_levels
        value = round(val_idx/(num_label_levels-1), 2)
        indices = np.where(data[:, att_idx] == value)[0]
        q_indices = set([a_to_q_indices[a] for a in indices])
        q_indices = [i for i in q_indices if i not in seen_indices]
        eval_indices.extend(q_indices[:num_of_each])
        train_indices.extend(q_indices[num_of_each:2*num_of_each])
        seen_indices.update(eval_indices)
        seen_indices.update(train_indices)

    dataset = np.array(dataset)
    train_data = list(dataset[train_indices])
    eval_data = list(dataset[eval_indices])
    return list(train_data), list(eval_data)
