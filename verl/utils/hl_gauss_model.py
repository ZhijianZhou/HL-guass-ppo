import dataclasses
from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from verl.utils.transformers_compat import get_auto_model_for_vision2seq

AutoModelForVision2Seq = get_auto_model_for_vision2seq()


@dataclasses.dataclass
class HLGaussOutput:
    logits: torch.Tensor
    hl_gauss_logits: torch.Tensor


def _init_hl_gauss_head_weights(v_head: nn.Linear, init_strategy: str = "normal", initializer_range: float = 0.02):
    if init_strategy == "normal":
        v_head.weight.data.normal_(mean=0.0, std=initializer_range)
    elif init_strategy == "zero":
        v_head.weight.data.zero_()
    elif init_strategy == "xavier":
        nn.init.xavier_normal_(v_head.weight.data)
    else:
        raise ValueError(f"Unsupported HL-Gauss init strategy: {init_strategy}")

    if v_head.bias is not None:
        v_head.bias.data.zero_()


def load_hl_gauss_valuehead_model(
    local_path,
    torch_dtype,
    model_config,
    trust_remote_code,
    n_bins: int = 101,
    v_min: float = -1.0,
    v_max: float = 1.0,
    sigma: Optional[float] = None,
    v_head_init_strategy: str = "normal",
    v_head_initializer_range: float = 0.02,
    dropout_prob: float = 0.1,
):
    if n_bins < 3:
        raise ValueError(f"n_bins must be >= 3, got: {n_bins}")
    if v_min >= v_max:
        raise ValueError(f"v_min ({v_min}) must be less than v_max ({v_max})")

    if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
        module_class = AutoModelForVision2Seq
    else:
        module_class = AutoModelForCausalLM

    model = module_class.from_pretrained(
        pretrained_model_name_or_path=local_path,
        torch_dtype=torch_dtype,
        config=model_config,
        attn_implementation="flash_attention_2",
        trust_remote_code=trust_remote_code,
    )

    hidden_size = model_config.hidden_size
    device = next(model.parameters()).device
    bin_width = (v_max - v_min) / n_bins
    if sigma is None:
        sigma = bin_width * 0.75

    model._hl_gauss_config = {
        "n_bins": n_bins,
        "v_min": v_min,
        "v_max": v_max,
        "bin_width": bin_width,
        "sigma": sigma,
    }

    model.v_head = nn.Linear(hidden_size, n_bins, bias=True).to(dtype=torch_dtype, device=device)
    _init_hl_gauss_head_weights(model.v_head, v_head_init_strategy, v_head_initializer_range)
    model.value_dropout = nn.Dropout(p=dropout_prob)
    model.hl_gauss_logits = None

    def hl_gauss_forward(self, *args, **kwargs):
        kwargs_with_hidden = dict(kwargs)
        kwargs_with_hidden["output_hidden_states"] = True

        model_outputs = self.model(*args, **kwargs_with_hidden)
        hidden_states = model_outputs.last_hidden_state if hasattr(model_outputs, "last_hidden_state") else model_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        hl_gauss_logits = self.v_head(self.value_dropout(hidden_states))
        self.hl_gauss_logits = hl_gauss_logits
        return HLGaussOutput(logits=lm_logits, hl_gauss_logits=hl_gauss_logits)

    import types as _types

    model.forward = _types.MethodType(hl_gauss_forward, model)
    return model
