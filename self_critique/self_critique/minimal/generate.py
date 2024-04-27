# sourcery skip: require-parameter-annotation
# ruff: noqa: ERA001
import copy
import inspect
from typing import Any

import torch
from transformers.generation.utils import GenerateOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


@torch.no_grad()
def generate(model, **kwargs: Any) -> GenerateOutput | torch.LongTensor:
    r"""
    Generates sequences of token ids for models with a language modeling head.

    Parameters:
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

    model_input_name = model.main_input_name
    inputs_tensor = model_kwargs.pop(model_input_name)
    batch_size = inputs_tensor.shape[0]

    model_kwargs = _prepare_encoder_decoder_kwargs_for_generation(
        model, inputs_tensor, model_kwargs
    )

    input_ids = (
        torch.ones((batch_size, 1), dtype=torch.long, device=inputs_tensor.device)
        * generation_config.decoder_start_token_id
    )

    generation_config.max_length = (
        generation_config.max_new_tokens + input_ids.shape[-1]
    )

    return model._greedy_search(
        input_ids,
        pad_token_id=generation_config.pad_token_id,
        output_scores=generation_config.output_scores,
        output_logits=generation_config.output_logits,
        return_dict_in_generate=True,
        **model_kwargs,
    )


def _prepare_encoder_decoder_kwargs_for_generation(
    model, inputs_tensor: torch.Tensor, model_kwargs: dict[str, Any]
) -> dict[str, Any]:
    encoder = model.get_encoder()

    encoder_signature = set(inspect.signature(encoder.forward).parameters)
    encoder_kwargs = {
        argument: value
        for argument, value in model_kwargs.items()
        if argument in encoder_signature
    }

    encoder_kwargs["return_dict"] = True
    encoder_kwargs[model.main_input_name] = inputs_tensor
    model_kwargs["encoder_outputs"] = encoder(**encoder_kwargs)

    return model_kwargs
