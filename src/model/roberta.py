
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
import torch.nn as nn


class RobertaForFirstCharPrediction(RobertaPreTrainedModel):
    """
    References:
        https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py
    """
    def __init__(self, config):
        super().__init__(config)
        
        # If add_pooling_layer is `True`, this will add a dense layer and a `tanh` activation.
        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, 29)
        # -> 0~25: alphabet, 26: digit, 27: punctuation, 28: exception

        self.post_init()
    
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        masked_word_labels=None,
        return_dict=None,
        **kwargs
    ):
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] # -> (bs, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output) # -> (bs, seq_len, hidden_size)
        logits = self.dense(sequence_output) # -> (bs, seq_len, 29)

        loss = None
        if masked_word_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 29), masked_word_labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class RobertaForNCharsPrediction(RobertaPreTrainedModel):
    """
    References:
        https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py
    """
    def __init__(self, config):
        super().__init__(config)
        
        # If add_pooling_layer is `True`, this will add a dense layer and a `tanh` activation.
        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.num_char_class)
        self.num_char_class = config.num_char_class

        self.post_init()
    
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        masked_word_labels=None,
        return_dict=None,
        **kwargs
    ):
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] # -> (bs, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output) # -> (bs, seq_len, hidden_size)
        logits = self.dense(sequence_output) # -> (bs, seq_len, 29)

        loss = None
        if masked_word_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_char_class), masked_word_labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
