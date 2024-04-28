# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Source: https://github.com/huggingface/transformers/blob/9fe3f585bb4ea29f209dc705d269fbe292e1128f/src/transformers/generation/utils.py#L1
# Refactored down to only the essentials I need.

import copy
from typing import Any

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    **kwargs: Any,
) -> torch.LongTensor:
    """Generates sequences of token ids for models with a language modeling head.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        kwargs (`dict[str, Any]`, *optional*):
            Ad hoc parametrization of `generation_config` and/or additional
            model-specific kwargs that will be forwarded to the `forward` function of
            the model. If the model is an encoder-decoder model, encoder specific kwargs
            should not be prefixed and decoder specific kwargs should be prefixed with
            *decoder_*.

    Return:
        [`~generation.GenerateEncoderDecoderOutput`]
    """
    generation_config = copy.deepcopy(model.generation_config)
    model_kwargs = generation_config.update(**kwargs)

    model_kwargs["encoder_outputs"] = model.get_encoder()(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )

    batch_size = input_ids.shape[0]
    input_ids = (
        torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)
        * generation_config.decoder_start_token_id
    )

    generation_config.max_length = (
        generation_config.max_new_tokens + input_ids.shape[-1]
    )

    return _greedy_search(
        model,
        input_ids,
        pad_token_id=generation_config.pad_token_id,
        attention_mask=attention_mask,
        **model_kwargs,
    )


def _greedy_search(
    model: PreTrainedModel,
    input_ids: torch.LongTensor,
    pad_token_id: int,
    **model_kwargs,
) -> torch.LongTensor:
    """Generates sequences of token ids using **greedy decoding**.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the
            `forward` function of the model. If model is an encoder-decoder model the
            kwargs should include `encoder_outputs`.

    Return:
        `torch.LongTensor`
    """
    eos_token_id = torch.tensor([model.generation_config.eos_token_id])

    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )

    finished = False
    while not finished:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=model.generation_config.output_attentions,
        )

        # the new tokens are the last item of the logits in the seq_len dimension
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
            1 - unfinished_sequences
        )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs["past_key_values"] = outputs.past_key_values

        unfinished_sequences = unfinished_sequences & ~_eos_token_stop_criteria(
            input_ids, eos_token_id
        )
        finished = unfinished_sequences.max() == 0

    return input_ids


def _eos_token_stop_criteria(
    input_ids: torch.LongTensor, eos_token_id: torch.LongTensor
) -> torch.BoolTensor:
    """Check if the last token is the EOS token."""
    if input_ids.device.type == "mps":
        # https://github.com/pytorch/pytorch/issues/77764#issuecomment-2067838075
        return (
            input_ids[:, -1]
            .tile(eos_token_id.shape[0], 1)
            .eq(eos_token_id.unsqueeze(1).to(input_ids.device))
            .sum(dim=0)
            .bool()
            .squeeze()
        )
    else:
        return torch.isin(input_ids[:, -1], eos_token_id.to(input_ids.device))
