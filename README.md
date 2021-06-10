# GLSearch
This repository is the PyTorch implementation of "GLSearch: Maximum Common Subgraph Detection via Learning to Search" (ICML2021). For more details, please refer to our paper.

## Installation

Package requirements are in requirements.txt. Please install required packages and versions.

## Run

To evaluate GLSearch on scalable datasets, please replace `config.py` with `config_scalable.py`. 

To evaluate GLSearch on datasets with less than ten thousande nodes, please replace `config.py` with `config_large.py`. 

Datasets are specified in `config.py`.

Once `config.py` is correctly set, run the below command:
```
python3 main.py
```
