import torch

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
  AutoModel, 
  AutoModelForTokenClassification,
  AutoConfig,
  PretrainedConfig
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.modeling_utils import PreTrainedModel

from typing import Dict, List, Optional, Set, Tuple, Union

# Class implemented guided by discussion at https://stackoverflow.com/questions/72503309/save-a-bert-model-with-custom-forward-function-and-heads-on-hugginface
class LinearModelConfig(PretrainedConfig):
    model_type="LinearModel"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# Implemented from https://jovian.com/rajbsangani/emotion-tuned-sarcasm
# However, used PreTrainedModel instead of nn.Module
class LinearModel(PreTrainedModel):
    config_class = LinearModelConfig
    def __init__(self, 
                 config): 
        #super(LinearModel,self).__init__() 
        super().__init__(config)
        #super(LinearModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.checkpoint = config.checkpoint
        #if "distilroberta" in self.checkpoint:
            #classifier_dropout = (
            #config.classifier_dropout if config.classifier_dropout is not None else config.#hidden_dropout_prob
        #)
        #if "distilbert" in self.checkpoint:
        #    classifier_dropout = config.dropout
        classifier_dropout = config.dropout
        #Load Model with given checkpoint and extract its body
        print(self.checkpoint)
        self.bert_config = AutoConfig.from_pretrained(self.checkpoint)

        self.model = AutoModel.from_pretrained(
            self.checkpoint,
            num_labels=config.num_labels
        )
        self.dropout = nn.Dropout(classifier_dropout) 
        self.classifier = nn.Linear(
            self.bert_config.hidden_size,
            config.num_labels
        ) # load and initialize weights

        #self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #Extract outputs from the body
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        #Add custom layers
        sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

        #logits = self.classifier(sequence_output[:,0,:].view(-1,768)) 
        logits = self.classifier(sequence_output) # calculate losses
        loss = self._return_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, 
            logits=logits, 
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
    def _return_loss(self, labels, logits):
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), 
                labels.view(-1)
            )
        return loss