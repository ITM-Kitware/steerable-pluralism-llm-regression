import logging

from src.algorithms.abstracts import ADM 
from src.utils.alignment_utils import select_response_index
from src.utils.KaleidoSys import KaleidoSys

log = logging.getLogger(__name__)

class KaleidoADM(ADM):
    def __init__(self,
                 model_name,
                 device='cuda',
                 **kwargs):
        self.system = KaleidoSys(model_name=model_name, device=device)

    def choose_response(self,
                        prompt,
                        responses,
                        alignment_target,
                        **kwargs):
        # Lists for saving output
        predicted_scores =  [{} for i in range(len(responses))]
        reasonings = ['' for i in range(len(responses))]
        # Loop over attribtues in target
        for i in range(len(responses)):
            response = responses[i]
            for attribute in list(alignment_target.keys()):
                statement = f"{prompt} {response}"
                # Get scores
                supports, opposes, either = self.system.get_valence(statement, 'Value', attribute)
                score = float(1.0*supports + 0.5*either + 0.0*opposes)
                predicted_scores[i][attribute] = score
                # Get reasoning
                reasoning = self.system.get_explanation(statement, 'Value', attribute)
                reasonings[i] += f"{reasoning} "
                
        # Select response using predicted values
        selected_response_index = select_response_index(predicted_scores, alignment_target)

        reasoning = reasonings[selected_response_index]
        choice_info = {'predictions': predicted_scores}
        
        log.info(f'Selected Response:\n{responses[selected_response_index]}\n')
        log.info(f'Reasoning:\n{reasoning}\n')

        return selected_response_index, reasoning, choice_info