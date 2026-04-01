#!/usr/bin/env bash
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-1}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-Math-7B"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/dapo/qwen2.5-7b-hl-gauss"}

project_name=${PROJECT_NAME:-dapo}
experiment_name=${EXPERIMENT_NAME:-qwen2.5-7b-hl-gauss}

adv_estimator=gae
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

actor_lr=1e-6
critic_lr=1e-5
gae_gamma=1.0
gae_lam=1.0

n_bins=101
v_min=-0.1
v_max=1.1
sigma=0.024

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 3))

enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 1))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

train_prompt_bsz=512
gen_prompt_bsz=$((train_prompt_bsz * 2))
n_resp_per_prompt=16
val_n_samples=${VAL_N_SAMPLES:-32}
train_prompt_mini_bsz=32

temperature=1.0
top_p=1.0
top_k=-1
val_top_p=1.0
val_temperature=0.65
critic_warmup=30

gen_tp=1
sp_size=1
use_dynamic_bsz=True
offload=True

actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 1))
critic_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 4))
log_prob_max_token_len_per_gpu=$((actor_max_token_len_per_gpu * 4))

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.return_raw_chat=True \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n_samples} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    +critic.use_hl_gauss=True \
    +critic.use_hl_gauss_critic=True \
    +critic.n_bins=${n_bins} \
    +critic.v_min=${v_min} \
    +critic.v_max=${v_max} \
    +critic.sigma=${sigma} \
    critic.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    critic.ppo_epochs=1 \
    critic.optim.lr=${critic_lr} \
    critic.model.path="${MODEL_PATH}" \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_max_token_len_per_gpu=${critic_max_token_len_per_gpu} \
    critic.forward_max_token_len_per_gpu=${critic_max_token_len_per_gpu} \
    critic.ulysses_sequence_parallel_size=${sp_size} \
    critic.use_dynamic_bsz=True \
    critic.model.fsdp_config.param_offload=${offload} \
    critic.model.fsdp_config.optimizer_offload=${offload} \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.critic_warmup=${critic_warmup} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=40 \
    trainer.total_epochs=50 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    "$@"
