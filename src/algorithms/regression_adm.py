import logging
import torch
import json
from jsonschema import validate
import outlines
from outlines.samplers import MultinomialSampler
import jinja2

from src.algorithms.abstracts import ADM 
from src.utils.alignment_utils import select_response_index
from src.prompt_engineering.utils import run_in_batches
from src.prompt_engineering.outlines_prompts import (
    regression_system_prompt,
    regression_prompt,
    regression_json_schema
)


log = logging.getLogger(__name__)

class RegressionADM(ADM):
    def __init__(self,
                 model_name,
                 device='auto',
                 sampler=MultinomialSampler(),
                 **kwargs):
        model_kwargs = kwargs.get('model_kwargs', {})
        if 'precision' in kwargs:
            if kwargs['precision'] == 'half':
                torch_dtype = torch.float16
            elif kwargs['precision'] == 'full':
                torch_dtype = torch.float32
            else:
                raise RuntimeError(
                    f"Unexpected value for 'precision' ({kwargs['precision']})"
                    ", expecting either 'half' or 'full'")

            model_kwargs['torch_dtype'] = torch_dtype

        self.model = outlines.models.transformers(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=kwargs.get('tokenizer_kwargs', {}))

        self.sampler = sampler
    
    def dialog_to_prompt(self, dialog):
        tokenizer = self.model.tokenizer.tokenizer

        try:
            encoded_dialog = tokenizer.apply_chat_template(dialog)
        except jinja2.exceptions.TemplateError:
            # Assume that the tokenizer chat template doesn't accept
            # system messages; combine system message first user
            # message
            system_msg, user_msg, *rest = dialog

            assert user_msg['role'] == 'user'

            updated_content = system_msg['content'] + '\n' + user_msg['content']

            dialog = [{'role': 'user', 'content': updated_content}, *rest]

            encoded_dialog = tokenizer.apply_chat_template(dialog)

        return tokenizer.decode(encoded_dialog)


    def choose_response(self,
                        prompt,
                        responses,
                        alignment_target,
                        num_samples=1,
                        generator_batch_size=10,
                        attribute_descriptions='src/prompt_engineering/attribute_descriptions.json',
                        attribute_scale_factor=100,
                        **kwargs):
        # Load attribute descriptions
        with open(attribute_descriptions) as f:
            att_descriptions = json.load(f)

        # Lists for saving output
        predicted_scores =  [{} for i in range(len(responses))]
        reasonings = ['' for i in range(len(responses))]

        # Loop over attribtues in target
        for attribute in list(alignment_target.keys()):
            # Create dialogs 
            system_prompt = regression_system_prompt(attribute,
                                            att_descriptions[attribute],
                                            attribute_scale_factor)
            dialogs = []
            # Sample multiple outputs
            for _ in range(num_samples):
                for response in responses:
                    dialog = [{'role': 'system', 'content': system_prompt}]
                    base_prompt = regression_prompt(attribute, prompt, response)
                    dialog.append({'role': 'user', 'content': base_prompt})
                    dialogs.append(dialog)

            # Set generator for structured output
            generator = outlines.generate.json(
                        self.model,
                        regression_json_schema(attribute_scale_factor),
                        sampler=self.sampler,
                        whitespace_pattern=r"[ ]?")

            # Covert dialog to text
            dialog_texts = [self.dialog_to_prompt(d) for d in dialogs]
            # Get output response(s)
            outputs = run_in_batches(generator, dialog_texts, generator_batch_size)

            # Log
            for i in range(num_samples):
                log.info("[bold]*DIALOG PROMPT*[/bold]",
                    extra={"markup": True})
                log.info(dialog_texts[i])
                log.info("[bold]*ADM RESPONSE*[/bold]",
                    extra={"markup": True})
                log.info(outputs[i])

            # Reshape to matrix of num_samples x len(responses)
            outputs = [outputs[i:i+len(responses)] for i in range(0,len(outputs),len(responses))]
            # Add to predicted scores and reasonings 
            for response_index in range(len(responses)):
                # get avaerage of predicted scores
                sum_predicted_scores = sum([outputs[i][response_index]['score'] for i in range(num_samples)])
                average_predicted_score = (sum_predicted_scores/num_samples) / attribute_scale_factor
                predicted_scores[response_index][attribute] = average_predicted_score
                # use first sampled reasoning
                reasonings[response_index] += f'{outputs[0][response_index]['reasoning']} ' 

        # Select response using predicted values
        selected_response_index = select_response_index(predicted_scores, alignment_target)

        reasoning = reasonings[selected_response_index]
        choice_info = {'predictions': predicted_scores}
        
        log.info(f'Selected Response:\n{responses[selected_response_index]}\n')
        log.info(f'Reasoning:\n{reasoning}\n')

        return selected_response_index, reasoning, choice_info