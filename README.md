# GLSearch
This repository is the PyTorch implementation of "GLSearch: Maximum Common Subgraph Detection via Learning to Search" (ICML2021). For more details, please refer to our paper.

## Installation

Package requirements are in `requirements.txt`. Please install required packages and versions.

Mark `/model/OurMCS/` and `/src/` as the source directories.

## Datasets

Place the required dataset `*.klepto` files under a new directory `/save/OurModelData/` (ex. `/save/OurModelData/aids700nef_..._None.klepto`).

## Run

### Testing

Select 1 of 3 desired dataset on `lines 62-77` of `model/OurMCS/config.py` by commenting out the rest.

Run the below command:
```
python3 main.py
```
### Training

Replace `config.py` with `config_training.py`.

Run the below command:
```
python3 main.py
```
