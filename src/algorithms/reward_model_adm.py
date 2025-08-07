import logging
import torch
from outlines.samplers import MultinomialSampler
from transformers import AutoTokenizer, pipeline

from src.algorithms.abstracts import ADM 


log = logging.getLogger(__name__)

class RewardModelADM(ADM):
    def __init__(self,
                 model_name,
                 device='auto',
                 sampler=MultinomialSampler(),
                 **kwargs):
        model_kwargs = kwargs.get('model_kwargs', {})
        print(model_name)
        self.rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = 0 # accelerator.device
        self.rm_pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            tokenizer=self.rm_tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )
        self.pipe_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": 1
        }

    def choose_response(self,
                        prompt,
                        responses,
                        alignment_target,
                        **kwargs):
        scores = []
        for response in responses:
            chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
            ]

            test_texts = [self.rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(self.rm_tokenizer.bos_token, "")]
            pipe_outputs = self.rm_pipe(test_texts, **self.pipe_kwargs)
            reward = [output[0]["score"] for output in pipe_outputs][0]
            scores.append(reward)

        selected_response_index = scores.index(max(scores))
        reasoning = 'Reward model scored it the highest.'
        choice_info = {'scores':scores}

        log.info(f'Selected Response:\n{responses[selected_response_index]}\n')
        log.info(f'Reasoning:\n{reasoning}\n')

        return selected_response_index, reasoning, choice_info