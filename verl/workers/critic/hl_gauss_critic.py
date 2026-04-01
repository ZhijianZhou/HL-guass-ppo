import logging
import math
import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_hl_gauss_value_loss
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.model import extract_multi_modal_inputs
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic.dp_critic import DataParallelPPOCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class HLGaussDataParallelPPOCritic(DataParallelPPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config, critic_module=critic_module, critic_optimizer=critic_optimizer)
        self.n_bins = config.get("n_bins", 101)
        self.v_min = config.get("v_min", -1.0)
        self.v_max = config.get("v_max", 1.0)
        self.bin_width = (self.v_max - self.v_min) / self.n_bins
        self.sigma = config.get("sigma", None) or (self.bin_width * 0.75)
        self.bin_centers = self.v_min + (torch.arange(self.n_bins) + 0.5) * self.bin_width

    def _distribution_to_scalar(self, distributions: torch.Tensor) -> torch.Tensor:
        bin_centers = self.bin_centers.to(distributions.device, dtype=distributions.dtype)
        return (distributions * bin_centers).sum(dim=-1)

    def _forward_micro_batch(self, micro_batch):
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits_rmpad = output.hl_gauss_logits.squeeze(0)

                if self.ulysses_sequence_parallel_size > 1:
                    logits_rmpad = gather_outputs_and_unpad(
                        logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                logits = pad_input(logits_rmpad, indices=indices, batch=batch, seqlen=seqlen)
                logits = logits[:, -response_length - 1 : -1, :]
            else:
                output = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits = output.hl_gauss_logits[:, -response_length - 1 : -1, :]

            probs = F.softmax(logits, dim=-1)
            values = self._distribution_to_scalar(probs)
            return values, logits

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        if not torch.isfinite(grad_norm):
            logger.warning(f"grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="hl_gauss critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = (
            ["responses", "input_ids", "response_mask", "attention_mask", "position_ids"]
            if "response_mask" in data.batch
            else ["responses", "input_ids", "attention_mask", "position_ids"]
        )
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                values, _ = self._forward_micro_batch(model_inputs)
            values_lst.append(values)

        values = torch.concat(values_lst, dim=0)
        if use_dynamic_bsz:
            values = restore_dynamic_batch(values, batch_idx_list)

        if "response_mask" in data.batch:
            response_mask = data.batch["response_mask"].to(values.device)
            values = values * response_mask
        return values

    @GPUMemoryLogger(role="hl_gauss critic", logger=logger)
    def update_critic(self, data: DataProto):
        self.critic_module.train()
        metrics = {"critic/vf_loss": 0.0}

        select_keys = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids", "values", "returns"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.critic_optimizer.zero_grad()
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    returns = model_inputs["returns"]

                    _, predicted_logits = self._forward_micro_batch(model_inputs)
                    probs = F.softmax(predicted_logits, dim=-1)
                    predicted_values = self._distribution_to_scalar(probs)

                    vf_loss = compute_hl_gauss_value_loss(
                        predicted_logits=predicted_logits,
                        returns=returns,
                        response_mask=response_mask,
                        n_bins=self.n_bins,
                        v_min=self.v_min,
                        v_max=self.v_max,
                        sigma=self.sigma,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        loss = vf_loss * loss_scale_factor
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                        loss = vf_loss * loss_scale_factor

                    loss.backward()

                    value_error = torch.abs(predicted_values - returns)
                    log_probs = F.log_softmax(predicted_logits, dim=-1)
                    entropy = masked_mean((-(probs * log_probs).sum(dim=-1)), response_mask)
                    clip_rate = masked_mean(((returns < self.v_min) | (returns > self.v_max)).float(), response_mask)

                    micro_batch_metrics.update(
                        {
                            "critic/hl_gauss_entropy": entropy.detach().item(),
                            "critic/hl_gauss_value_error": masked_mean(value_error, response_mask).detach().item(),
                            "critic/hl_gauss_clip_rate": clip_rate.detach().item(),
                            "critic/hl_gauss_entropy_warn_low": float(entropy < 0.5),
                            "critic/hl_gauss_entropy_warn_high": float(entropy > math.log(self.n_bins) * 0.8),
                        }
                    )

                    metrics["critic/vf_loss"] += vf_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"critic/grad_norm": grad_norm.detach().item()})

        self.critic_optimizer.zero_grad()
        return metrics
