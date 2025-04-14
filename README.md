# SilVar-Med: A Speech-Driven Visual Language Model for Explainable Abnormality Detection in Medical Imaging


## Installation

```bash
conda create -n SilVarMed python=3.10.13
conda activate SilVarMed
git clone https://github.com/Hanhpt23/SilVarMed.git
cd SilVarMed
pip install -r requirements.txt
```


In this work, we focus on 2 critical problems in the medical domain which are using speech to interact with vision language model and providing explanation for each prediction.



Training
Training configuration:
      model:
            vision_model: [here](train_configs/train_abnormal_OmniMedVQA_llama3.1.yaml#L7) at Line 7
            audio_model: [here](train_configs/train_abnormal_OmniMedVQA_llama3.1.yaml#L8) at Line 8
            language_model: [here](train_configs/train_abnormal_OmniMedVQA_llama3.1.yaml#L9) at Line 9
            others: lora_r, freeze_vision, freeze_audio
            - If you want to train the model end-to-end, set `freeze_vision` and `freeze_audio` to `False` [here](train_configs/train.yaml#L17) on lines 17 and 18
      datasets:
            batch_size: [here](train_configs/train_abnormal_OmniMedVQA_llama3.1.yaml#L22) at Line 22
            image_path: [here](train_configs/train_abnormal_OmniMedVQA_llama3.1.yaml#L35) at Line 35
            ann_path: [here](train_configs/train_abnormal_OmniMedVQA_llama3.1.yaml#L36) at Line 36
            audio_path: [here](train_configs/train_abnormal_OmniMedVQA_llama3.1.yaml#L37) at Line 37
      run:
            max_epoch: [here](train_configs/train_abnormal_OmniMedVQA_llama3.1.yaml#L47) at Line 47
            iters_per_epoch: [here](train_configs/train_abnormal_OmniMedVQA_llama3.1.yaml#L50) at Line 50
            output_dir: [here](train_configs/train_abnormal_OmniMedVQA_llama3.1.yaml#L53) at Line 53
            wandb_token: [here](train_configs/train_abnormal_OmniMedVQA_llama3.1.yaml#L67) at Line 67

Testing configuration:
      model:
            vision_model: [here](eval_configs/eval_abnormal_OmniMedVQA_llama3.1.yaml#L7) at Line 7
            audio_model: [here](eval_configs/eval_abnormal_OmniMedVQA_llama3.1.yaml#L8) at Line 8
            language_model: [here](eval_configs/eval_abnormal_OmniMedVQA_llama3.1.yaml#L9) at Line 9
            ckpt: [here](eval_configs/eval_abnormal_OmniMedVQA_llama3.1.yaml#L10) at Line 10

      evaluation_datasets:
            eval_file_path: 
            batch_size: [here](eval_configs/eval_abnormal_OmniMedVQA_llama3.1.yaml#L39) at Line 39
            img_path: [here](eval_configs/eval_abnormal_OmniMedVQA_llama3.1.yaml#L36) at Line 36
            audio_path: [here](eval_configs/eval_abnormal_OmniMedVQA_llama3.1.yaml#L38) at Line 38
      run:
            save_path: [here](eval_configs/eval_abnormal_OmniMedVQA_llama3.1.yaml#L47) at Line 47
            
```bash
torchrun --nproc_per_node 1 train.py \
      --cfg-path train_configs/train_abnormal.yaml\
      --cfg-eval-path eval_configs/eval_abnormal.yaml\
      --eval-dataset audio_val
```


Evaluation 
```bash
torchrun --nproc_per_node 2 --master_port=29501 evaluate.py \
      --cfg-path eval_configs/eval_abnormal_one_train.yaml\
      --eval-dataset audio_val
```
