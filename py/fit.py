import random
import sys
import torch
import wandb

from datasets import Dataset
from transformers import (
  AutoTokenizer,
  DataCollatorForTokenClassification
)
from utils import (
  compute_concise_metrics,
  compute_full_metrics,
  compute_CRF_metrics,
  load_id_and_label_files,
  load_model,
  set_wandb_params,
)
from train_pipes import *

random.seed(42)
set_wandb_params()

task = "ner" 
model_type = sys.argv[1]
model_checkpoint = ["distilroberta-base"]
hidden_size=[100, 200, 300, 400]
classification_dropout = [0.1, 0.2, 0.3]
batch_size = [4, 8, 16, 32]
if model_type == "crf-bilstm":
  bilstm_dropout = [0, 0.1, 0.2, 0.3]
  use_lstm=True
else:
  bilstm_dropout = [0]
  use_lstm=False

output_attentions=True
output_hidden_states=True 

label_list, label2id, id2label = load_id_and_label_files()

print(label_list)
train_tokenized_datasets = Dataset.load_from_disk(
  "data/for_models/train")
val_tokenized_datasets = Dataset.load_from_disk(
  "data/for_models/val")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

idx = 0
for hs in hidden_size:
  for d in bilstm_dropout:
    for cd in classification_dropout:
      for bs in batch_size:
        for ms in model_checkpoint:
          tokenizer = AutoTokenizer.from_pretrained(
            ms, 
            add_prefix_space=True
          )
          config = set_model_config(
            model_type=model_type,
            pretrained_model_name_or_path=ms,
            checkpoint = ms,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, 
            num_labels=len(label_list),
            label2id = label2id,
            id2label = id2label,
            dropout = cd,
            bilstm_dropout = d,
            use_lstm = use_lstm,
            hidden_size = hs,
            batch_size = bs
          )
          print(config)
          wandb.init(config=config)

          learning_rate = 1e-4
          weight_decay = 1e-5
          num_epochs = 5

          model = load_model(model_type, ms, config)
          data_collator = DataCollatorForTokenClassification(tokenizer)

          print("Fitting model using full training loop.")
          train_dataloader, eval_dataloader = data_loaders(
            train_tokenized_datasets, 
            val_tokenized_datasets,
            bs,
            data_collator
          )
              
          weight_decay=weight_decay
          adam_eps=1e-8

          num_update_steps_per_epoch = len(train_dataloader)
          num_training_steps = num_epochs * num_update_steps_per_epoch
          num_warmup_steps = round(num_training_steps*0.1)

          patience=1
          es_min_delta=0.001

          optimizer, lr_scheduler, es, num_training_steps = init_optimizers(
            model,
            learning_rate,
            weight_decay,
            adam_eps,
            num_update_steps_per_epoch,
            num_training_steps,
            num_warmup_steps,
            patience,
            es_min_delta
          )

          wandb.watch(model, log_freq=100)
          finetuned_model, overall_f1 = train_loop(
            model = model, 
            model_type = model_type,
            train_dataloader = train_dataloader, 
            eval_dataloader = eval_dataloader, 
            optimizer = optimizer, 
            scheduler = lr_scheduler,
            es = es,
            num_epochs = num_epochs, 
            label_names = label_names,
            num_training_steps = num_training_steps
          )

          if overall_f1  > 0.81 or idx == 0:
            finetuned_model.save_pretrained(f"models/ner-{model_type}-bs{bs}-{ms}-f1-{overall_f1}.model")

          wandb.finish()
          del model
          del finetuned_model
          del tokenizer
          del data_collator
          del train_dataloader
          del eval_dataloader
          idx += 1