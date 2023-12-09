# A Critical Toponymy Framework for NER

This repository contains the data and code to train and validate the models in the paper ["Toward a Critical Toponymy Framework for Named Entity Recognition: A Case Study of Airbnb in New York City"](https://aclanthology.org/2023.emnlp-main.284/), presented at EMNLP 2023. The benchmark DistilRoBERTa-CRF model from the paper can be downloaded from the [HuggingFace model hub](https://huggingface.co/maybemkl/distilroberta-crf-geo). 

## Training

To train a model from scratch, simply run:

```
python py/fit.py <MODEL_TYPE>
```

where `MODEL_TYPE` can either be `linear`, `crf` or `bilstm-crf`. The `.json` files in `data` contain the training, testing, and validation data, while the corresponding subfolders have the data preprocessed with the [`datasets`](https://huggingface.co/docs/datasets/index) library. `fit.py` requires the data in the latter format, but we are also providing the raw `.json` files to make the data more accessible and easy to use in other frameworks.

## Processing

TBA

## Analysis

For code to produce the paper visuals and outputs, please reach out to the authors.
