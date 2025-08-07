import random
import logging

from src.algorithms.abstracts import ADM 
from src.utils.alignment_utils import select_response_index


log = logging.getLogger(__name__)

class OracleADM(ADM):
    def __init__(self, **kwargs):
        pass

    def choose_response(self, prompt, responses, alignment_target, labels, **kwargs):
        # Get true values
        true_values = []
        for response_index in range(len(responses)):
            label = {}
            for attribute in alignment_target.keys():
                if attribute in labels[response_index]:
                    label[attribute] = labels[response_index][attribute]
                else:
                    label[attribute] = 0.5 # placeholder
            true_values.append(label)
        
        # Select response using true values
        selected_response_index = select_response_index(true_values, alignment_target)

        reasoning = 'Looked at scores.'
        choice_info ={}
        
        log.info(f'Prompt:\n{prompt}\n')
        log.info(f'Responses:\n{responses}\n')
        log.info(f'Selected Response:\n{responses[selected_response_index]}\n')
        log.info(f'Reasoning:\n{reasoning}\n')

        return selected_response_index, reasoning, choice_info