# GLSearch
This repository is the PyTorch implementation of "GLSearch: Maximum Common Subgraph Detection via Learning to Search" (ICML2021). For more details, please refer to our paper.

## Installation

Package requirements are in `requirements.txt`. Please install required packages and versions. We now provide a Dockerfile to make this setup easier.

Mark `/model/OurMCS/` and `/src/` as the source directories.

### Docker Setup

To build and run the Docker environment use the commands below:
```
cd /path/to/GLSearch
nvidia-docker build -t "glsearch" .
nvidia-docker run -e "HOSTNAME=$(cat /etc/hostname)" -v /path/to/GLSearch:/workspace -it glsearch bash
```

## Datasets

Get the datasets from: https://drive.google.com/drive/folders/1l7_WmO54gPHZ4YwzOqewZtZhWIxn44ZV?usp=sharing

Place the `*.klepto` files under a new directory `/save/OurModelData/` (ex. `/save/OurModelData/aids700nef_..._None.klepto`).

### Google Drive Filename Issues

We noticed Google Drive sometimes renames downloaded files, which will cause issues in data loading. Please check the filenames after downloading.

Data loading is done on `line 310` of `model/OurMCS/data_model.py` (as of commit `4e10d5f79e7f5bd53e74628f52619a0fc10f4c80`). You can check here whether `tp` is pointing to the correct file (NOTE: it is optional whether we include the `.klepto` extension in `tp`).

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
