import random
import logging
import torch
import json
from jsonschema import validate
from collections import Counter
import outlines
from outlines.samplers import MultinomialSampler
import jinja2

from src.algorithms.abstracts import ADM 
from src.utils.alignment_utils import select_response_index
from src.prompt_engineering.utils import (
    list_to_multiple_choice,
    run_in_batches
)
from src.prompt_engineering.outlines_prompts import (
    comparative_regression_system_prompt,
    comparative_regression_prompt,
    comparative_regression_json_schema
)


log = logging.getLogger(__name__)

class ComparativeRegressionADM(ADM):
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

    # Returns dialog with in-context learning examplea
    def get_icl_dialog(self, incontext_data, attribute, prompt_to_match, attribute_scale_factor):
        # Get all labeled examples for the attribute
        labeled_examples = []
        for probe in incontext_data:
            if attribute in probe['labels'][0]:
                responses = probe['responses']
                scores = [label[attribute] for label in probe['labels']]
                reasonings = [r_dict[attribute] for r_dict in probe['reasonings']]
                multiple_choices = list_to_multiple_choice(responses)
                example_prompt = comparative_regression_prompt(attribute, probe['prompt'], multiple_choices)
                # Create example output
                example_output = {}
                for letter, response in multiple_choices.items():
                    response_index = responses.index(response)
                    example_output[letter] = {
                        'reasoning':reasonings[response_index],
                        'score':scores[response_index]*attribute_scale_factor
                        }
                # # Validate response against schema
                # correct_schema = json.loads(comparative_regression_json_schema(list(multiple_choices.keys()), 
                # validate(instance=example_output, schema=correct_schema)
                labeled_examples.append({'prompt':example_prompt,'output':example_output})

        # Sort via bert similarity
        possible_prompts = [icl_example["prompt"] for icl_example in labeled_examples]
        from bert_score import score
        _, _, F1 = score([prompt_to_match]*len(possible_prompts), possible_prompts, lang="en")
        sorted_F1, low_to_high = zip(*sorted(zip(F1, labeled_examples), key=lambda x: x[0]))
        sorted_labeled_examples = low_to_high[::-1]
        
        # Add an example of each score 
        scores = {entry['output'][key]['score'] for entry in labeled_examples for key in entry['output']}
        selected_icl_examples = []
        for s in sorted(scores):
            for example in sorted_labeled_examples:
                if any(r['score'] == s for r in example['output'].values()):
                    if example not in selected_icl_examples:
                        selected_icl_examples.append(example)
                    break

        # Format example dialogs
        icl_dialogs = []
        for example in selected_icl_examples:
            icl_dialogs.extend([
                {"role": "user", "content": example['prompt']},
                {"role": "assistant", "content": f'{example["output"]}'}
            ])
        return icl_dialogs
    
    def choose_response(self,
                        prompt,
                        responses,
                        alignment_target,
                        num_samples=1,
                        generator_batch_size=10,
                        shuffle_responses=True,
                        attribute_descriptions='src/prompt_engineering/attribute_descriptions.json',
                        attribute_scale_factor=100,
                        fewshot=False,
                        **kwargs):
        # Load attribute descriptions
        with open(attribute_descriptions) as f:
            att_descriptions = json.load(f)

        # Load incontext data
        if fewshot:
            incontext_file = kwargs.get('incontext_file', '')
            with open(incontext_file) as f:
                incontext_data = json.load(f)
            
        # Lists for saving output
        predicted_scores =  [{} for i in range(len(responses))]
        reasonings = ['' for i in range(len(responses))]

        # Loop over attribtues in target
        for attribute in list(alignment_target.keys()):
            # Create dialogs 
            system_prompt = comparative_regression_system_prompt(attribute,
                                                                 att_descriptions[attribute],
                                                                 attribute_scale_factor)
            dialogs = []
            mc_dictionaries = []
            # Sample multiple outputs
            for _ in range(num_samples):
                dialog = [{'role': 'system', 'content': system_prompt}]
                if shuffle_responses:
                    shuffled_responses = random.sample(responses, len(responses))
                    multiple_choice_options = list_to_multiple_choice(shuffled_responses)
                else:
                    multiple_choice_options = list_to_multiple_choice(responses)
                base_prompt = comparative_regression_prompt(attribute, prompt, multiple_choice_options)
                if fewshot:
                    icl_dialog = self.get_icl_dialog(incontext_data, 
                                                        attribute, 
                                                        base_prompt,
                                                        attribute_scale_factor)
                    dialog.extend(icl_dialog)
                dialog.append({'role': 'user', 'content': base_prompt})

                mc_dictionaries.append(multiple_choice_options)
                dialogs.append(dialog)

            # Set generator for structured output
            generator = outlines.generate.json(
                        self.model,
                        comparative_regression_json_schema(list(multiple_choice_options.keys()), attribute_scale_factor),
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

            # Replace MC letters with response indexes
            for i in range(num_samples):
                letters = list(outputs[i].keys())
                for letter in letters:
                    response_index = responses.index(mc_dictionaries[i][letter])
                    outputs[i][response_index] = outputs[i].pop(letter)

            # Add to predicted scores and reasonings 
            for response_index in range(len(responses)):
                # get avaerage of predicted scores
                sum_predicted_scores = sum([outputs[i][response_index]['score'] for i in range(num_samples)])
                average_predicted_score = (sum_predicted_scores/num_samples) / attribute_scale_factor
                predicted_scores[response_index][attribute] = average_predicted_score
                # use first sampled reasoning
                reasonings[response_index] += f'{outputs[0][response_index]['reasoning']} '
        
        # Select response using true values
        selected_response_index = select_response_index(predicted_scores, alignment_target)

        reasoning = reasonings[selected_response_index]
        choice_info = {'predictions': predicted_scores}
        
        log.info(f'Selected Response:\n{responses[selected_response_index]}\n')
        log.info(f'Reasoning:\n{reasoning}\n')

        return selected_response_index, reasoning, choice_info