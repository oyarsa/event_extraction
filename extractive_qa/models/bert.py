# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2020 Guillaume Becquin.
# MODIFIED FOR CAUSE EFFECT EXTRACTION
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

from torch import nn
from transformers import BertModel, BertPreTrainedModel


class BertForCauseEffect(BertPreTrainedModel):
    def __init__(self, config):
        assert config.num_labels == 2

        super().__init__(config)

        self.bert = BertModel(config)
        self.cause_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.effect_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        cause_start_positions=None,
        cause_end_positions=None,
        effect_start_positions=None,
        effect_end_positions=None,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = bert_output[0]  # (bs, max_query_len, dim)
        cause_logits = self.cause_outputs(hidden_states)  # (bs, max_query_len, 2)
        effect_logits = self.effect_outputs(hidden_states)  # (bs, max_query_len, 2)
        cause_start_logits, cause_end_logits = cause_logits.split(1, dim=-1)
        effect_start_logits, effect_end_logits = effect_logits.split(1, dim=-1)

        cause_start_logits = cause_start_logits.squeeze(-1)  # (bs, max_query_len)
        cause_end_logits = cause_end_logits.squeeze(-1)  # (bs, max_query_len)
        effect_start_logits = effect_start_logits.squeeze(-1)  # (bs, max_query_len)
        effect_end_logits = effect_end_logits.squeeze(-1)  # (bs, max_query_len)

        outputs = (
            cause_start_logits,
            cause_end_logits,
            effect_start_logits,
            effect_end_logits,
        ) + bert_output[2:]
        if (
            cause_start_positions is not None
            and cause_end_positions is not None
            and effect_start_positions is not None
            and effect_end_positions is not None
        ):
            # sometimes the start/end positions are outside our model inputs, we
            # ignore these terms
            ignored_index = cause_start_logits.size(1)
            cause_start_positions.clamp_(0, ignored_index)
            cause_end_positions.clamp_(0, ignored_index)
            effect_start_positions.clamp_(0, ignored_index)
            effect_end_positions.clamp_(0, ignored_index)

            loss_fn = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_cause_loss = loss_fn(cause_start_logits, cause_start_positions)
            end_cause_loss = loss_fn(cause_end_logits, cause_end_positions)
            start_effect_loss = loss_fn(effect_start_logits, effect_start_positions)
            end_effect_loss = loss_fn(effect_end_logits, effect_end_positions)

            avg_loss = (
                start_cause_loss + end_cause_loss + start_effect_loss + end_effect_loss
            ) / 4
            outputs = (avg_loss,) + outputs

        # (loss), start_cause_logits, end_cause_logits, start_effect_logits,
        # end_effect_logits, (hidden_states), (attentions)
        return outputs
