import numpy as np

def value_dict_to_list(alignment_target, value_dict):
    values = []
    for attribute in alignment_target.keys():
        values.append(value_dict[attribute])
    return values

def euclidean_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1) - np.array(vector2))

def select_response_index(predicted_values, alignment_target):
    if not set(alignment_target.keys()) <= set(predicted_values[0].keys()):
        raise RuntimeError("Attributes in target and predicted values do not match.")
    target_values = value_dict_to_list(alignment_target, alignment_target)
    best_index = 0
    min_dist = np.inf
    for i in range(len(predicted_values)):
        response_values = value_dict_to_list(alignment_target, predicted_values[i])
        dist = euclidean_distance(target_values, response_values)
        if dist < min_dist:
            min_dist = dist 
            best_index = i
    return best_index