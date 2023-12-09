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

For code to reproduce the paper visuals and downstream data outputs, please reach out to the authors.

## Citations

If you use our models or data, please use the following citation:

```
@inproceedings{brunila-etal-2023-toward,
    title = "Toward a Critical Toponymy Framework for Named Entity Recognition: A Case Study of Airbnb in {N}ew {Y}ork City",
    author = "Brunila, Mikael  and
      LaViolette, Jack  and
      CH-Wang, Sky  and
      Verma, Priyanka  and
      F{\'e}r{\'e}, Clara  and
      McKenzie, Grant",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.284",
    pages = "4676--4695",
    abstract = "Critical toponymy examines the dynamics of power, capital, and resistance through place names and the sites to which they refer. Studies here have traditionally focused on the semantic content of toponyms and the top-down institutional processes that produce them. However, they have generally ignored the ways in which toponyms are used by ordinary people in everyday discourse, as well as the other strategies of geospatial description that accompany and contextualize toponymic reference. Here, we develop computational methods to measure how cultural and economic capital shape the ways in which people refer to places, through a novel annotated dataset of 47,440 New York City Airbnb listings from the 2010s. Building on this dataset, we introduce a new named entity recognition (NER) model able to identify important discourse categories integral to the characterization of place. Our findings point toward new directions for critical toponymy and to a range of previously understudied linguistic signals relevant to research on neighborhood status, housing and tourism markets, and gentrification.",
}
```
