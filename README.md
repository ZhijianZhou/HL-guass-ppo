# HL-Gauss PPO on verl

This repository is a local adaptation of `verl` for running a minimal HL-Gauss critic baseline on DAPO-style PPO training.

The main goal of this fork is to migrate the legacy `hl_guass` experiment setting onto a newer `verl` codebase while keeping the change set small and focused on one runnable baseline.

## What Is Included

This repository currently contains the minimal HL-Gauss pipeline needed for:

- an HL-Gauss histogram value head
- an HL-Gauss critic implementation
- an HL-Gauss value loss
- FSDP worker integration for selecting the HL-Gauss critic
- a minimal DAPO recipe script for `Qwen2.5-Math-7B`

## Key Files

Runtime integration:

- `verl/workers/config/critic.py`
- `verl/utils/hl_gauss_model.py`
- `verl/workers/critic/hl_gauss_critic.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/workers/fsdp_workers.py`
- `verl/workers/critic/__init__.py`

Recipe:

- `recipe/dapo/run_qwen2-7b_hl_guass.sh`

## HL-Gauss Summary

HL-Gauss replaces scalar value regression with histogram-based value prediction.

Instead of predicting a single value directly, the critic predicts logits over `n_bins` value bins. The training target is a Gaussian-smoothed distribution over those bins, and the scalar value used by PPO is recovered from the predicted distribution expectation.

The minimal baseline in this repo uses:

- `n_bins=101`
- `v_min=-0.1`
- `v_max=1.1`
- `sigma=0.024`
- `adv_estimator=gae`

## Minimal Baseline Script

The entry script is:

```bash
recipe/dapo/run_qwen2-7b_hl_guass.sh
```

It uses:

- `python3 -m verl.trainer.main_ppo`
- `reward.reward_manager.name=dapo`
- `+critic.use_hl_gauss=True`
- `+critic.use_hl_gauss_critic=True`

Default paths in the script:

- `MODEL_PATH=${HOME}/verl/models/Qwen2.5-Math-7B`
- `TRAIN_FILE=${HOME}/verl/data/dapo-math-17k.parquet`
- `TEST_FILE=${HOME}/verl/data/aime-2024.parquet`

## Example Run

```bash
bash recipe/dapo/run_qwen2-7b_hl_guass.sh
```

You can override the defaults from the shell, for example:

```bash
NUM_GPUS=8 \
NNODES=1 \
MODEL_PATH=/path/to/Qwen2.5-Math-7B \
TRAIN_FILE=/path/to/dapo-math-17k.parquet \
TEST_FILE=/path/to/aime-2024.parquet \
bash recipe/dapo/run_qwen2-7b_hl_guass.sh
```

## Current Scope

This repo currently only targets the minimal HL-Gauss baseline.

Not included yet:

- variance-weighted HL-Gauss
- turn-level HL-Gauss critic
- mean-pooled HL-Gauss critic
- broader `retool` or `searchr1` HL-Gauss variants

## Notes

- The script name keeps the legacy spelling `hl_guass` for compatibility with earlier experiments.
- The actual method name is written as `HL-Gauss` in the code and documentation.
- The current codebase uses newer Python syntax in upstream files, so a Python `3.10+` environment is recommended.

## Goal of This Fork

This fork is intended to be a clean place to iterate on HL-Gauss PPO experiments on top of modern `verl`, starting from one minimal runnable DAPO recipe and extending from there.