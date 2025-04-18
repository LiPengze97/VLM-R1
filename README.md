# CVD_Reasoning_MAI

---

## Installation

Create and activate the conda environment:

```bash
conda create -n vlm-r1 python=3.10
conda activate vlm-r1
```

Then run the setup script:

```bash
bash setup.sh
```

---

## Data Location

ECG image data archive is located at:

```
/dgx1data/aii/tao/m327768/ecg_files/MAI_CVD_train_ECG_figs_one_patient.tar.gz
```

---

## Weights & Biases (wandb)

After installing `wandb` during setup, log into your account using:

```bash
wandb login
```

You will be prompted to enter your API key, which can be found on your [W&B account settings page](https://wandb.ai/settings). Once logged in, you'll be ready to track your machine learning experiments.

---

## Data Preparation

Before training, make sure to copy and extract the training, validation, and test data. The data archive is located at:

```
/dgx1data/aii/tao/m327768/train.jsonl
```

To extract it into the proper directory:

```bash
tar -xzvf train_data_json.tar.gz -C CVD_R_MAI/src/train_data
```

---

## SLURM Environment Setup

Load the required module:

```bash
module nccl2-cuda12.1-gcc11/2.18.3
```

---

## GRPO Training

To start training with GRPO:

```bash
bash run_scripts/run_grpo_cvd.sh
```

Please make sure you have updated the contents of the .sh file accordingly.

Training logs will be saved to:

```
runs
```

Additional training metrics and visualizations will be available in your Weights & Biases dashboard.
