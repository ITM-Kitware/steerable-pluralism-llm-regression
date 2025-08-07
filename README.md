# Steerable Pluralism: Pluralistic Alignment via Few-Shot Comparative Regression
Github repo corresponding to our AIES 2025 Paper: "Steerable Pluralism: Pluralistic Alignment via Few-Shot Comparative Regression."
It includes reframing of the MIC and Helpsteer2 datasets as steerable benchmarks and the implementation of our few-shot comparative regression-based alignment for LLM decison making.

Paper: TODO

### Set-Up
We reccomend using a virtual Python environment. Dependencies can be installed by calling:
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
The reformatted HelpSteer2 data is in `data/helpsteer2/`. This was created via `load_helpsteer2.py` with few-shot reasoning statement created using `generate_helpsteer2_reasoning.py`.

#### Alignment Targets
Alignment targets are in `alignment_targets/`. The sampled subsets were selected using `sample_targets.py`.

### Running a steerable pluralistic model / automatic decision maker (ADM)
Experiment config files are in: `src/configs/experiment/`
Model config files are in: `src/configs/adm/`

To run an experiment, call `run_experiment.py` with the experiment config, ADM, and alignment target:
```
python run_experiment.py +experiment=mic/deterministic_llama adm=proposed_spm alignment_target=mic/target_high
```
This will output a log and `input_output.json` file to the `outputs/` directory. The `hydra.run.dir` parameter can be used to specify a different output directory.

### Scoring
To score, call `score.py` with the alignment target and `input_output.json` file:
```
python score.py --target alignment_targets/mic/target_high.json --input_outputs outputs/mic_proposed_spm/input_output.json
```
## Citation
If you find this work useful, please consider citing our paper:
```
TODO
```
## Disclaimer
We emphasize that our work should be considered academic research, as we cannot fully guarantee model outputs are free of inaccuracies or biases that may pose risks if relied upon for medical decision-making. Please consult a qualified healthcare professional for personal medical needs.

This material is based upon work supported by the Defense Advanced Research Projects Agency and the Air Force Research Laboratory, contract number(s): FA8650-23-C-7316. Any opinions, findings and conclusions, or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of AFRL or DARPA.