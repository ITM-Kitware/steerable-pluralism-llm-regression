# Steerable Pluralism: Pluralistic Alignment via Few-Shot Comparative Regression
Github repo corresponding to our AIES 2025 paper: "Steerable Pluralism: Pluralistic Alignment via Few-Shot Comparative Regression."
This repo includes the reframing of the MIC and Helpsteer2 datasets as steerable benchmarks and the implementation of our few-shot comparative regression-based alignment approach for LLM decision making.

### Set-Up
We recommend using a virtual Python environment to manage dependencies. Python dependencies can be installed by calling:
```
python -m pip install -r requirements.txt
```

### Steerable Benchmark Data

#### MIC
To reformat the MIC dataset as a steerable benchmark, first complete the user agreement and download the data from: [https://github.com/SALT-NLP/mic](https://github.com/SALT-NLP/mic).

Place the `MIC.csv` file in the `data/mic/` folder, then from the `data/` directory run:
```
python load_mic.py
```
#### HelpSteer2
The reformatted HelpSteer2 data is provided in `data/helpsteer2/`. This was created via `load_helpsteer2.py` with few-shot reasoning statement created using `generate_helpsteer2_reasoning.py`.

#### Alignment Targets
Alignment targets are in `alignment_targets/`. The sampled subsets were selected using `sample_targets.py`.

### Running an automatic decision maker (ADM) model
We utilize the [hydra](https://hydra.cc/) framework to define parameter configurations for experiments and models. 
- Experiment config files are in: `src/configs/experiment/`
- Model config files are in: `src/configs/adm/`

To run an experiment, call `run_experiment.py` with the experiment config, ADM config, and alignment target. For example:
```
python run_experiment.py +experiment=mic/deterministic_llama adm=proposed_spm alignment_target=mic/target_high
```
This will output a log and `input_output.json` file to the `outputs/` directory. The `hydra.run.dir` parameter can be used to specify a different output directory.

### Scoring
To calculate an alignment score, call `score.py` with the alignment target and `input_output.json` file. For example:
```
python score.py --target alignment_targets/mic/target_high.json --input_outputs outputs/mic_proposed_spm/input_output.json
```
## Citation
If you find this work useful, please consider citing our paper:
```
@inproceedings{Adams_etal25AIES,
  title={Steerable Pluralism: Pluralistic Alignment via Few-Shot Comparative Regression},
  author={Jadie Adams and Brian Hu and Emily Veenhuis and David Joy and Bharadwaj Ravichandran and Aaron Bray and Anthony Hoogs and Arslan Basharat}
  booktitle={Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society},
  volume={8},
  year={2025}
}
```
## Disclaimer
We emphasize that our work should be considered academic research, as we cannot fully guarantee model outputs are free of inaccuracies or biases that may pose risks if relied upon for medical decision-making. Please consult a qualified healthcare professional for personal medical needs.

This material is based upon work supported by the Defense Advanced Research Projects Agency and the Air Force Research Laboratory, contract number(s): FA8650-23-C-7316. Any opinions, findings and conclusions, or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of AFRL or DARPA.
