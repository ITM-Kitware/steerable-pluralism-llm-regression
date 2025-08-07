import random
import logging

from src.algorithms.abstracts import ADM 


log = logging.getLogger(__name__)

class RandomADM(ADM):
    def __init__(self, **kwargs):
        pass

    def choose_response(self, prompt, responses, alignment_target, **kwargs):
        selected_response_index = random.randint(0, len(responses)-1)
        reasoning = 'Selected randomly.'
        choice_info ={}
        
        log.info(f'Prompt:\n{prompt}\n')
        log.info(f'Responses:\n{responses}\n')
        log.info(f'Selected Response:\n{responses[selected_response_index]}\n')
        log.info(f'Reasoning:\n{reasoning}\n')

        return selected_response_index, reasoning, choice_info