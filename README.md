# SilVar-Med: A Speech-Driven Visual Language Model for Explainable Abnormality Detection in Medical Imaging


## Installation

```bash
conda create -n SilVarMed python=3.10.13
conda activate SilVarMed
git clone https://github.com/Hanhpt23/SilVarMed.git
cd SilVarMed
pip install -r requirements.txt
```

<!-- 
## Training
### Visual encoder and audio encoder setting
We have released our checkpoint [here](https://drive.google.com/file/d/1nYrygg9O4NmaxIptW_nQCyrPoP58U-RK/view?usp=drive_link), you can download and use it as a pretrained weight or for inference.

### Training Configuration
- Set the pretrained checkpoint for downstream tasks [here](train_configs/train.yaml#L10) at Line 10.
- Set the training image path [here](train_configs/train.yaml#L35) at Line 35
- Set the training annotation path [here](train_configs/train.yaml#L36) at Line 36
- Set the training audio path [here](train_configs/train.yaml#L37) at Line 37
- Set the output directory [here](train_configs/train.yaml#L54) at Line 54
- Set the wandb token [here](train_configs/train.yaml#L69) at Line 69
- If you want to train the model end-to-end, set `freeze_vision` and `freeze_audio` to `False` [here](train_configs/train.yaml#L17) on lines 17 and 18


### Evaluation Configuration
- Set the checkpoint [here](eval_configs/evaluate.yaml#L10) at Line 10.
- Set the evaluation image path [here](eval_configs/evaluate.yaml#L36) at Line 36
- Set the evaluation annotation path [here](eval_configs/evaluate.yaml#L35) at Line 35
- Set the evaluation audio path [here](eval_configs/evaluate.yaml#L38) at Line 38
- Set the output directory [here](eval_configs/evaluate.yaml#L54) at Line 54

### Run
- To run on a terminal:

```bash
torchrun --nproc_per_node 2 train.py \
        --cfg-path train_configs/train.yaml\
        --cfg-eval-path eval_configs/evaluate.yaml\
        --eval-dataset audio_val
```

- To submit to an HPC:
```bash
sbatch scripts/OmniMod/train.sh
```

## Evaluation
- To run on a terminal:
```bash
torchrun --nproc_per_node 2 evaluate.py \
      --cfg-path eval_configs/evaluate.yaml\
      --eval-dataset audio_val
```

- To submit to an HPC:
```bash
sbatch scripts/OmniMod/evaluate.sh
```

## Dataset structure
```
OmniMod
├── train
│   ├── audio
│   ├── images
│   ├── train.json
├── test
│   ├── audio
│   ├── images
│   ├── test.json

└── pretrained_checkpoint
    └── checkpoint_19.pth
```

#### Structure of `OmniMod_sets.json`
```
[
      {
            "query": "",
            "outputs": "",
            "image": ""
      },
      ...
]
``` -->
