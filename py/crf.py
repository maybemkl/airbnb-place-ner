import torch

from torch import nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF #https://github.com/kmkurn/pytorch-crf
from transformers import (
  AutoModel, 
  AutoModelForTokenClassification,
  AutoConfig,
  PretrainedConfig
)

from mcrf import MaskedCRF

from transformers.modeling_utils import PreTrainedModel

from typing import Dict, List, Optional, Set, Tuple, Union

class CRFModelConfig(PretrainedConfig):
    model_type="CRFModel"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# Expanding on https://github.com/shushanxingzhe/transformers_ner/blob/main/models.py
class CRFModel(PreTrainedModel):
    config_class = CRFModelConfig
    def __init__(self, 
                 config): 
        #super(CRFModel,self).__init__() 
        super().__init__(config)
        self.config = config
        #print(config)
        self.num_labels = config.num_labels
        self.checkpoint = config.checkpoint
        self.masked = config.masked
        #if "distilroberta" in config.checkpoint:
        #    classifier_dropout = (
        #        config.classifier_dropout if config.classifier_dropout is not None else config.#hidden_dropout_prob
        #)
        #if "distilbert" in config.checkpoint:
        #    classifier_dropout = config.dropout
        classifier_dropout = config.dropout
        #Load Model with given checkpoint and extract its body
        self.bert_config = AutoConfig.from_pretrained(self.checkpoint)
    
        self.model = AutoModel.from_pretrained(
            self.checkpoint,
            num_labels=config.num_labels
        )
        self.dropout = nn.Dropout(classifier_dropout) 
        if self.config.use_lstm:
            self.bilstm = nn.LSTM(
                #input_size=self.config.hidden_size, 
                input_size=self.bert_config.hidden_size,
                hidden_size=self.config.hidden_size // 2,
                #(config.hidden_size) // 2, 
                # This isn't a good value here, need to fix
                # dropout=classifier_dropout, 
                #dropout=self.config.bilstm_dropout, 
                dropout = 0.2,
                batch_first=True,
                bidirectional=True
            )
        if not self.masked:
            self.crf = CRF(
                num_tags=config.num_labels, 
                batch_first=True
            )
        else:
            self.crf = MaskedCRF(
                num_tags=config.num_labels, 
                batch_first=True,
                constraints=config.constraints,
                masked_training=True
            )
        if self.config.use_lstm:
            self.classifier = nn.Linear(
                in_features=self.config.hidden_size,
                out_features=config.num_labels
            ) # load and initialize weights
        else:
            self.classifier = nn.Linear(
                in_features=self.bert_config.hidden_size,
                out_features=config.num_labels
            ) # load and initialize weights

        #self.init_weights()

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
        # Make sure this is actually the last hidden state
        # and not the pooled output
        # Make sure these are actually the embeddings
        sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state
        if self.config.use_lstm:
            sequence_output, hc = self.bilstm(sequence_output)
        # Any other way to get the logits? Why do we need a linear layer?
        logits = self.classifier(sequence_output) # calculate losses
        loss, tags = self._return_loss(labels, logits, attention_mask)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return loss, tags

    def _return_loss(self, labels, logits, attention_mask):
        if labels is not None:
            masked_labels, mask_bool = self._create_mask(labels)
            log_likelihood = self.crf(logits, masked_labels)
            tags = self.crf.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits)
            tags = self._restore_masked_tags(tags, mask_bool)
            print("tags", tags)
        return loss, tags

    def _restore_masked_tags(self, tags, mask_bool):
        tags = torch.Tensor(tags)
        tags[mask_bool == False] = -100
        return tags

    # Drops columns with -100 values
    def _create_mask(self, labels):
        mask = labels.detach().clone()
        mask[labels == -100] = 0
        mask[labels != -100] = 1
        masked_labels = labels*mask.int()
        mask_bool = mask.type(torch.BoolTensor)
        return masked_labels, mask_bool