# A Critical Toponymy Framework for NER

This repository contains the data and code to train and validate the models in the paper ["Toward a Critical Toponymy Framework for Named Entity Recognition: A Case Study of Airbnb in New York City"](https://aclanthology.org/2023.emnlp-main.284/), presented at EMNLP 2023. The benchmark DistilRoBERTa-CRF model from the paper can be downloaded from the [HuggingFace model hub](https://huggingface.co/maybemkl/distilroberta-crf-geo). For code to produce the paper visuals and outputs, please reach out to the authors.

To train a model from scratch, simply run:

```
python py/fit.py <MODEL_TYPE>
```

where `MODEL_TYPE` can either be `linear`, `crf` or `bilstm-crf`
