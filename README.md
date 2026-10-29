# Interpretable Difficulty-Aware Knowledge Tracing in Tutor-Student Dialogues

Official implementation of **Interpretable Difficulty-Aware Knowledge Tracing in Tutor-Student Dialogues**.

> Shuyan Huang, Alexander Scarlatos, Jaewook Lee, and Andrew Lan  
> University of Massachusetts Amherst  
> [Paper](https://arxiv.org/abs/2605.01097) | [PDF](https://arxiv.org/pdf/2605.01097)

## Overview

This project performs turn-level knowledge tracing (KT) in tutor-student dialogues. Given the original problem, the dialogue history, and the next tutor-posed task, the model predicts whether the student will answer correctly.

The proposed framework has three components:

1. A **knowledge estimator** uses an LLM to estimate the student's current ability from the original question and dialogue history.
2. A **difficulty estimator** uses the original question, dialogue context, and current tutor task to estimate task difficulty.
3. An **IRT-based predictor** combines ability and difficulty with a learnable discrimination scalar to produce an interpretable correctness probability.

The implementation also includes LLMKT and deep-learning KT baselines (DKT, DKVMN, SAINT, AKT, and simpleKT).

## Results

Turn-level correctness prediction reported in the paper (percent):

| Model | QATD2k Accuracy | QATD2k AUC | MathDial Accuracy | MathDial AUC |
|---|---:|---:|---:|---:|
| LLMKT | 64.24 | 64.89 | 68.64 | 75.99 |
| **Difficulty-Aware DialogueKT** | **64.29** | **65.25** | **68.82** | **76.59** |

The repository uses `eedi` as the command-line dataset identifier for QATD2k.

## Repository structure

```text
.
├── data/annotated/       # Included QATD2k/Eedi annotations and KC dictionary
├── dialogue_kt/          # Data loading, prompting, models, training, and evaluation
│   └── models/           # LLMKT and neural KT model implementations
├── results/              # Example evaluation outputs
├── run.sh                # Example training command
├── evaluate.sh           # Example evaluation command
└── requirements.txt
```

## Setup

Python 3.10 is recommended. Create an isolated environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The paper's experiments use `meta-llama/Meta-Llama-3.1-8B-Instruct`, LoRA fine-tuning, BFloat16 mixed precision, and a single NVIDIA L40 GPU. Before training, obtain access to the gated model on Hugging Face and authenticate locally:

```bash
huggingface-cli login
```

Hardware requirements depend on the base model and whether quantization is enabled. Use `--quantize True` to reduce GPU memory usage where supported.

## Data

The repository includes the processed QATD2k splits used by the `eedi` configuration:

```text
data/annotated/eedi_train_mathdial_format.csv
data/annotated/eedi_valid_mathdial_format.csv
data/annotated/eedi_test_mathdial_format.csv
data/annotated/kc_dict_eedi_mathdial_format.json
```

Each annotated dialogue contains the dialogue turns, metadata, response-correctness labels, and knowledge-component (KC) annotations. The code also supports `mathdial` and `comta`; their source/annotated files are not distributed in this repository and must be prepared at the paths expected by `dialogue_kt/data_loading.py`.

## Training

Replace `MODEL_PATH` with a local path or Hugging Face model identifier:

```bash
python -m dialogue_kt.main train \
  --dataset eedi \
  --tag_src mathdial_format \
  --base_model MODEL_PATH \
  --model_type lmkt \
  --model_name difficulty_aware_dialogkt \
  --use_irt True \
  --batch_size 1 \
  --grad_accum_steps 64 \
  --lr 2e-4 \
  --epochs 5 \
  --r 16 \
  --lora_alpha 16 \
  --use_best_auc True
```

The defaults for the LLM-based model match the main setup: 5 epochs, learning rate `2e-4`, LoRA rank 16, LoRA scaling factor 16, effective batch size 64, weight decay `1e-2`, and gradient clipping at 1.0. Checkpoints are written to `saved_models/`.

To enable Weights & Biases logging, set `WANDB_API_KEY` and add `--use_wandb True --wandb_project <project-name>`.

The provided shell script can also be used after replacing its `model_path` placeholder:

```bash
bash run.sh
```

## Evaluation

Use the same model name, base model, fold, and IRT setting used during training:

```bash
python -m dialogue_kt.main test \
  --dataset eedi \
  --tag_src mathdial_format \
  --base_model MODEL_PATH \
  --model_type lmkt \
  --model_name difficulty_aware_dialogkt \
  --use_irt True
```

Metrics and qualitative outputs are saved under `results/`. The paper reports Accuracy and AUC while excluding the first labeled turn of each dialogue, for which no prior observation is available.

For a quick pipeline check on a small subset, append `--debug True` to the training or evaluation command. This verifies code flow but does not reproduce the reported results.

## Baselines and useful options

- Set `--model_type` to `dkt`, `dkvmn`, `saint`, `akt`, or `simplekt` for neural KT baselines.
- Add `--hyperparam_sweep` during training to run the built-in search.
- Add `--crossval` for cross-validation and `--fold <N>` for a specific fold.
- Set `--quantize True` to load the LLM with 4-bit quantization.
- Set `--pack_kcs False` to use the unpacked KC prompting path.
- Run `python -m dialogue_kt.main --help` or a subcommand with `--help` for the full option list.

## Citation

If you use this code, please cite:

```bibtex
@article{huang2026interpretable,
  title={Interpretable Difficulty-Aware Knowledge Tracing in Tutor-Student Dialogues},
  author={Huang, Shuyan and Scarlatos, Alexander and Lee, Jaewook and Lan, Andrew},
  journal={arXiv preprint arXiv:2605.01097},
  year={2026}
}
```

## Acknowledgments

This work is partially supported by Renaissance Philanthropy under the Learning Engineering Virtual Institute (LEVI) initiative and the National Science Foundation under grant 2237676.
