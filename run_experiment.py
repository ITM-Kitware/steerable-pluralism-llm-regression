import json
from copy import deepcopy
import os
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base=None,
            config_path="src/configs",
            config_name="base")

def main(cfg: DictConfig) -> None:
    cfg = instantiate(cfg, recursive=True)
    adm = cfg.adm.instance

    # Using the hydra generated output directory for the run
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Set output paths 
    logfile_path = None
    if cfg.save_log:
        logfile_path = os.path.join(output_dir, "adm.log")
    logging.basicConfig(filename=logfile_path, level=logging.INFO)

    save_input_output_to_path = None
    if cfg.save_input_output:
        save_input_output_to_path = os.path.join(output_dir, "input_output.json")

    # Get alignment target
    with open(os.path.join('alignment_targets', f'{cfg.alignment_target}.json')) as f:
        alignment_target = json.load(f)
    log.info(f'Aligning to target:\n{alignment_target}\n')

    if cfg.get('force_determinism', False):
        import torch
        torch_seed = 0
        log.info(f"Setting `torch.manual_seed` to: {torch_seed}")
        torch.manual_seed(torch_seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        log.info("Setting `torch_use_deterministic_algorithms` to True")
        torch.use_deterministic_algorithms(
            cfg.get('torch_use_deterministic_algorithms', True),
            warn_only=True)
        import random
        random_seed = 0
        log.info(f"Setting `random.seed` to: {random_seed}")
        random.seed(random_seed)
        import numpy as np
        numpy_random_seed = 0
        log.info(f"Setting `numpy.random.seed` to: {numpy_random_seed}")
        np.random.seed(numpy_random_seed)
        
    # Load model
    log.info('Loading ADM.')
    if hasattr(adm, 'load_model'):
        adm.load_model()

    # Capture inputs and outputs in a similar format to what's used by
    # our internal evaluation framework code
    inputs_outputs = []
    for dataset in cfg.datasets:
        log.info(f'Running {dataset}')
        with open(os.path.join('data', dataset, 'eval.json')) as f:
            scenarios = json.load(f)
        
        # Get incotext learning file
        inference_kwargs = cfg.adm.get('inference_kwargs', {})
        if 'fewshot' in inference_kwargs and inference_kwargs.fewshot:
            cfg.adm.inference_kwargs.incontext_file = os.path.join('data', dataset, 'train.json')
        
        for scenario in scenarios:
            # pass labels to oracle
            if cfg.oracle: 
                labels = deepcopy(scenario['labels'])
                selected_response_index, reasoning, choice_info = adm.choose_response(scenario['prompt'],
                                                                    scenario['responses'],
                                                                    alignment_target,
                                                                    labels,
                                                                    **cfg.adm.get('inference_kwargs', {}))
            else: 
                selected_response_index, reasoning, choice_info = adm.choose_response(scenario['prompt'],
                                                                    scenario['responses'],
                                                                    alignment_target,
                                                                    **cfg.adm.get('inference_kwargs', {}))
            output = {}
            output['selected_response_index'] = selected_response_index
            output['reasoning'] = reasoning
            output['choice_info'] = choice_info
            inputs_outputs.append({'input':scenario, 'output':output})
            
            # Save input output 
            if save_input_output_to_path is not None:
                with open(save_input_output_to_path, 'w') as f:
                    json.dump(inputs_outputs, f, indent=2)

    log.info("Done.")


if __name__ == "__main__":
    main()