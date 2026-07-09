# Start Classifying: Categorical Critics for LLM Reinforcement Learning

This repository contains the code for **Start Classifying: Categorical Critics
for LLM Reinforcement Learning**, accepted to **CoLM 2026**.

The project studies a simple change to PPO-style reinforcement learning for
large language models: replace the standard scalar MSE value critic with a
categorical critic trained by cross-entropy. Our main method, **HL-Gauss PPO**,
keeps the actor objective and advantage computation unchanged, but trains the
critic to predict a distribution over discretized value bins.

The paper and BibTeX will be added later.

## Overview

In reinforcement learning with verifiable rewards (RLVR), rewards are often
sparse and binary. A scalar MSE critic compresses this naturally bimodal return
structure into a single mean, which can distort advantage estimates. HL-Gauss
PPO instead represents value targets categorically, using Gaussian-smoothed
target distributions over a fixed value support.

HL-Gauss PPO is designed as a drop-in change that can:

- replace standard PPO critics with minimal changes;
- provide richer supervision than scalar value regression;
- produce more symmetric and lower-variance advantage signals;
- improve over strong PPO-based baselines and DAPO on mathematical reasoning
  and agentic Search-R1 benchmarks.

## Repository Layout

```text
code/verl/      Forked verl training stack with custom critic implementations
code/recipe/    Launch scripts and Hydra overrides for PPO/DAPO experiments
```

## Key Implementation Files

The HL-Gauss critic is implemented as a model head plus a worker-side critic:

- `code/verl/utils/hl_gauss_model.py`
- `code/verl/workers/critic/hl_gauss_critic.py`
- `code/verl/workers/fsdp_workers.py`

Representative launch scripts include:

- `code/recipe/dapo/run_qwen2.5-7b-math_hl-guass.sh`
- `code/recipe/searchr1/run_qwen2-7b_ppo_hl_guass.sh`
- `code/recipe/retool/run_qwen2-7b_hl_guass.sh`

Several script names preserve the historical spelling `guass` for compatibility
with existing experiment paths and glob patterns.

## Running Experiments

The training code is launched from source rather than as an installed package.
Most experiments follow this pattern:

```bash
cd code
export PYTHONPATH=$PWD:$PYTHONPATH
python3 -m recipe.dapo.main_dapo <hydra-overrides>
```

HL-Gauss-specific Hydra overrides typically include:

```bash
+critic.use_hl_gauss=True
+critic.use_hl_gauss_critic=True
+critic.n_bins=101
+critic.v_min=-0.1
+critic.v_max=1.1
+critic.sigma=0.009
```

The provided launch scripts encode the environment variables, NCCL settings,
data paths, and cluster-specific configuration used in our experiments. Many of
those paths point to internal training infrastructure, so treat the scripts as
reference configurations when adapting the code to a new environment.

## Status

This is an initial research code release. Some cluster paths and experiment
scripts may require local adaptation before they can be run outside the original
training environment.
