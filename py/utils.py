import json
import numpy as np
import os

from datasets import load_metric

from sklearn.metrics import (
  classification_report, 
  f1_score
)

from linear import LinearModel
from crf import CRFModel

metric = load_metric("seqeval")
with open('data/for_models/label_list', 'r') as f:
    label_names = f.read().split('\n')

def set_wandb_params():
  # set the wandb project where this run will be logged
  os.environ["WANDB_PROJECT"]="ner_airbnb"

  # save your trained model checkpoint to wandb
  os.environ["WANDB_LOG_MODEL"]="true"

  # turn off watch to log faster
  os.environ["WANDB_WATCH"]="false"

  #os.environ["WANDB_DISABLED"] = "true"

def load_model(model_type, model_checkpoint, config):
  if model_type == 'linear':
    print(f"Fitting {model_type} model")
    model = LinearModel( 
      config=config)#.to(device)
  if model_type == 'crf':
    print(f"Fitting {model_type} model")
    model = CRFModel(
      config=config)#.to(device)
  if model_type == 'crf_bilstm':
    print(f"Fitting {model_type} model")
    model = CRFModel(
      config=config)
  if model_type == 'mcrf':
    print(f"Fitting {model_type} model")
    model = CRFModel(
      config=config)
  return model

def load_id_and_label_files():
  with open('data/for_models/label_list', 'r') as f:
    label_list = f.read().split('\n')

  with open('data/for_models/label2id', 'r') as fp:
    label2id = json.load(fp)

  with open('data/for_models/id2label', 'r') as fp:
    id2label = json.load(fp)
    
  return label_list, label2id, id2label

def compute_metrics(eval_preds):
  logits, labels = eval_preds
  predictions = np.argmax(logits, axis=-1)
  
  # Remove ignored index (special tokens) and convert to labels
  true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
  true_predictions = [
      [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
      for prediction, label in zip(predictions, labels)
  ]
  all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
  return {
      "precision": all_metrics["overall_precision"],
      "recall": all_metrics["overall_recall"],
      "f1": all_metrics["overall_f1"],
      "accuracy": all_metrics["overall_accuracy"],
  }

def compute_concise_metrics(p):
  with open('data/for_models/label_list', 'r') as f:
    label_list = f.read().split('\n')
  
  predictions, labels = p
  predictions = np.argmax(predictions, axis=2)

  true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
  true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

  results = metric.compute(predictions=true_predictions, references=true_labels)
  return {"precision": results["overall_precision"], 
          "recall": results["overall_recall"], 
          "f1": results["overall_f1"], 
          "accuracy": results["overall_accuracy"]}
    
def compute_full_metrics(p):
  with open('data/for_models/label_list', 'r') as f:
    label_list = f.read().split('\n')
  
  predictions, labels = p
  predictions = np.argmax(predictions, axis=2)
    
  true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    
  true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

  results = metric.compute(predictions=true_predictions, references=true_labels)
  return(results)
#metric = evaluate.load("seqeval")

def compute_CRF_metrics(p):
  with open('data/for_models/label_list', 'r') as f:
    label_list = f.read().split('\n')
  
  labels = p.label_ids.tolist()
  preds = p.predictions.tolist()
  preds = [[int(x) for x in p] for p in preds]
  true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] 
                      for prediction, label in zip(preds, labels)]
  true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] 
                 for prediction, label in zip(preds, labels)]
  results = metric.compute(
    predictions=true_predictions, 
    references=true_labels
  )
  return(results)

# From https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
  def __init__(self, patience=1, min_delta=0):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.min_validation_loss = np.inf

  def early_stop(self, validation_loss):
    if validation_loss < self.min_validation_loss:
      self.min_validation_loss = validation_loss
      self.counter = 0
    elif validation_loss > (self.min_validation_loss + self.min_delta):
      self.counter += 1
      if self.counter >= self.patience:
        return True
    return False



