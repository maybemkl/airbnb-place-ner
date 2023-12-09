import evaluate
import torch
import wandb

from crf import (
  CRFModel,
  CRFModelConfig
)
from datasets import load_metric
from linear import (
  LinearModel, 
  LinearModelConfig
)
from mcrf import (
  allowed_transitions
)
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm
from transformers import (
  AdamW,
  AutoConfig,
  DataCollatorForTokenClassification,
  get_linear_schedule_with_warmup,
  Trainer,
  TrainingArguments
)
from typing import List
from utils import EarlyStopper

with open('data/for_models/label_list', 'r') as f:
    label_names = f.read().split('\n')

def register_model(model_type):
    # register your config and your model
    if model_type=='linear':
        AutoConfig.register("LinearModel", LinearModelConfig)
        AutoModel.register(LinearModelConfig, LinearModel)
    if model_type=='crf' or model_type=='crf_bilstm':
        AutoConfig.register("CRFModel", CRFModelConfig)
        AutoModel.register(CRFModelConfig, CRFModel)
    if model_type=='mcrf':
        AutoConfig.register("MCRFModel", MCRFModelConfig)
        AutoModel.register(MCRFModelConfig, MCRFModel)
        
def set_model_config(
    model_type, 
    pretrained_model_name_or_path,
    checkpoint,
    output_attentions,
    output_hidden_states, 
    num_labels,
    label2id,
    id2label,
    dropout,
    bilstm_dropout,
    use_lstm = False,
    hidden_size = 768,
    batch_size = 32,
    masked=False):
    
    if model_type=='linear':
        config = LinearModelConfig(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            checkpoint = checkpoint,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, 
            num_labels=num_labels,
            label2id = label2id,
            id2label = id2label,
            dropout = dropout,
            batch_size = batch_size,
            model_type = model_type
        )
    if model_type=='crf' or 'crf_bilstm':
        config = CRFModelConfig(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            checkpoint = checkpoint,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, 
            num_labels=num_labels,
            label2id = label2id,
            id2label = id2label,
            dropout = dropout,
            use_lstm = use_lstm,
            hidden_size = hidden_size,
            batch_size = batch_size,
            model_type = model_type,
            masked = masked
        )
        
    if model_type=='mcrf':
        #constraints = allowed_transitions('BIO', {int(k):v for k,v in id2label.items()})
        config = CRFModelConfig(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            checkpoint = checkpoint,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, 
            num_labels=num_labels,
            label2id = label2id,
            id2label = id2label,
            dropout = dropout,
            use_lstm = use_lstm,
            hidden_size = hidden_size,
            batch_size = batch_size,
            model_type = model_type,
            masked = True,
            constraints = id2label
        )
    return config

def data_loaders(
    train_tokenized_datasets, 
    val_tokenized_datasets,
    batch_size,
    data_collator):
    
    train_tokenized_datasets = train_tokenized_datasets.remove_columns(["tokens", "ner_tags"])
    val_tokenized_datasets = val_tokenized_datasets.remove_columns(["tokens", "ner_tags"])
    
    train_dataloader = DataLoader(
      train_tokenized_datasets, 
      shuffle=True, 
      batch_size=batch_size, 
      collate_fn=data_collator
    )
    
    eval_dataloader = DataLoader(
      val_tokenized_datasets, 
      batch_size=batch_size, 
      collate_fn=data_collator
    )
    return train_dataloader, eval_dataloader

def init_optimizers(
    model,
    lr,
    weight_decay,
    adam_eps,
    num_update_steps_per_epoch,
    num_training_steps,
    num_warmup_steps,
    patience,
    es_min_delta):
    
    optimizer = AdamW(
      model.parameters(), 
      lr=lr,
      weight_decay=weight_decay,
      eps=1e-8
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch = -1
    ) 

    es = EarlyStopper(
        patience=patience, 
        min_delta=es_min_delta
    )
    
    return optimizer, lr_scheduler, es, num_training_steps

def _postprocess_linear(logits, labels):
    predictions = logits.argmax(dim=-1)
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()
    return _true_preds_labels(predictions, labels)

def _postprocess_crf(predictions, labels):
    predictions = [[int(x) for x in p] for p in predictions]
    labels = labels.tolist()
    return _true_preds_labels(predictions, labels)

def _true_preds_labels(predictions, labels):
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
    [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
    ]
    return true_predictions, true_labels

def log_metrics(y_true, y_pred, tags):
    """
    Logs precision, recall, f1-score, and accuracy for each tag using Wandb.

    Args:
        y_true (List[List[int]]): Ground truth labels.
        y_pred (List[List[int]]): Predicted labels.
        tags (List[str]): List of all possible tags.

    Returns:
        None
    """

    # Calculate precision, recall, f1-score, and accuracy for each tag
    results = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(tags))))

    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Log metrics for each tag
    for i, tag in enumerate(tags):
        wandb.log({
            f'{tag}_accuracy': results[0][i],
            f'{tag}_precision': results[0][i],
            f'{tag}_recall': results[1][i],
            f'{tag}_f1_score': results[2][i],
            f'{tag}_support': results[3][i],
        })

    # Log overall accuracy
    wandb.log({
        'accuracy': accuracy,
    })

def train_loop(
  model, 
  model_type,
  train_dataloader, 
  eval_dataloader, 
  optimizer, 
  scheduler,
  es,
  num_epochs, 
  label_names: List[str],
  num_training_steps
):
  
  progress_bar_train = tqdm(range(num_training_steps))
  progress_bar_eval = tqdm(range(num_epochs * len(eval_dataloader)))
  
  metric = evaluate.load("seqeval")
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model.to(device)

  for epoch in range(num_epochs):
      model.train()
      for batch in train_dataloader:
          batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          if model_type=='linear':
            loss = outputs.loss
          else:
            loss = outputs[0]
          loss.backward()

          nn.utils.clip_grad_norm_(
              model.parameters(), 
              1.0
          )
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()
          wandb.log({"loss": loss})
          progress_bar_train.update(1)

      model.eval()
      for batch in eval_dataloader:
          batch = {k: v.to(device) for k, v in batch.items()}
          with torch.no_grad():
              outputs = model(**batch)
          
          if model_type=='linear':
            val_loss = outputs.loss
            true_predictions, true_labels = _postprocess_linear(
                logits = outputs.logits, 
                labels = batch["labels"]
            )
          else:
            val_loss = outputs[0]
            true_predictions, true_labels = _postprocess_crf(
                predictions = outputs[1], 
                labels = batch["labels"]
            )

          metric.add_batch(
              predictions=true_predictions, 
              references=true_labels
          )
          
          progress_bar_eval.update(1)
      
      results = metric.compute(zero_division=0)
      print(results)
      print(
      f"epoch {epoch}:",
          {
              key: results[f"overall_{key}"]
              for key in ["precision", "recall", "f1", "accuracy"]
          },
      )
      wandb.log({
          key: results[f"overall_{key}"]
          for key in ["precision", "recall", "f1", "accuracy"]
      })
      
      if es.early_stop(val_loss):
          print("Early stopping.")
          break
      else:
          print("No early stopping for epoch.")
  return model, results["overall_f1"]