import random
import logging
import torch
from collections import Counter
import outlines
from outlines.samplers import MultinomialSampler
import jinja2
import json
from jsonschema import validate

from src.algorithms.abstracts import ADM 
from src.utils.alignment_utils import select_response_index
from src.prompt_engineering.utils import (
    list_to_multiple_choice,
    run_in_batches
)
from src.prompt_engineering.outlines_prompts import (
    baseline_system_prompt,
    baseline_prompt,
    baseline_json_schema
)


log = logging.getLogger(__name__)

class BaselineADM(ADM):
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
    def get_icl_dialog(self, incontext_data, alignment_target, prompt_to_match):
        # Get all labeled examples for the attribute
        labeled_examples = []
        for probe in incontext_data:
            responses = probe['responses']
            multiple_choices = list_to_multiple_choice(responses)
            example_prompt = baseline_prompt(probe['prompt'], multiple_choices)
            # Get correct choice
            selected_response_index = select_response_index(probe['labels'], alignment_target)
            selected_response = responses[selected_response_index]
            for letter, response in multiple_choices.items():
                if response == selected_response:
                    correct_letter = letter
            # Create example output
            example_output = {}
            example_reasoning = ''
            reasoning_dict = probe['reasonings'][selected_response_index]
            for attribute in alignment_target:
                example_reasoning += reasoning_dict[attribute]
            example_output['reasoning'] = example_reasoning
            example_output['choice'] = correct_letter
            # Add to examples
            labeled_examples.append({'prompt':example_prompt,'output':example_output})

        # Sort via bert similarity
        possible_prompts = [icl_example["prompt"] for icl_example in labeled_examples]
        from bert_score import score
        _, _, F1 = score([prompt_to_match]*len(possible_prompts), possible_prompts, lang="en")
        sorted_F1, low_to_high = zip(*sorted(zip(F1, labeled_examples), key=lambda x: x[0]))
        sorted_labeled_examples = low_to_high[::-1]
        selected_icl_examples = sorted_labeled_examples[:5]
        
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
                        fewshot=False,
                        **kwargs):
        # Load incontext data
        if fewshot:
            incontext_file = kwargs.get('incontext_file', '')
            with open(incontext_file) as f:
                incontext_data = json.load(f)

        # Create dialogs 
        dialogs = []
        mc_dictionaries = []
        # Sample multiple outputs
        for _ in range(num_samples):
            dialog = [{'role': 'system', 'content': baseline_system_prompt(alignment_target)}]
            if shuffle_responses:
                shuffled_responses = random.sample(responses, len(responses))
                multiple_choice_options = list_to_multiple_choice(shuffled_responses)
            else:
                multiple_choice_options = list_to_multiple_choice(responses)
            base_prompt = baseline_prompt(prompt, multiple_choice_options)
            if fewshot:
                icl_dialog = self.get_icl_dialog(incontext_data, 
                                                    alignment_target, 
                                                    base_prompt)
                dialog.extend(icl_dialog)
            dialog.append({'role': 'user', 'content': base_prompt})

            mc_dictionaries.append(multiple_choice_options)
            dialogs.append(dialog)

        # Set generator for structured output
        generator = outlines.generate.json(
                    self.model,
                    baseline_json_schema(list(multiple_choice_options.keys())),
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
            letter = outputs[i]['choice']
            response_index = responses.index(mc_dictionaries[i][letter])
            outputs[i]['choice'] = response_index

        # Get most common response index
        counts = Counter([d['choice'] for d in outputs if 'choice' in d])
        selected_response_index = counts.most_common()[0][0]
        # Use reasoning of first output with most common choice
        reasoning = [d['reasoning'] for d in outputs if d['choice']==selected_response_index][0]
        # Add counts to choice info
        choice_info = {'selected_response_index_sample_counts':counts.most_common()}
        
        log.info(f'Selected Response:\n{responses[selected_response_index]}\n')
        log.info(f'Reasoning:\n{reasoning}\n')

        return selected_response_index, reasoning, choice_info