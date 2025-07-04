# WALLE-Affinity (WAFFLE)

This repository is the official implementation of **W**al**LE**-**AFF**inity (WAFFLE),
integrating both Graph and Protein Language information for antibody-antigen
affinity prediction framed as either a ranking or a regression tasks. The method is descrived in our
paper [AbRank: A Benchmark Dataset and Metric-Learning Framework for Antibody–Antigen Affinity Ranking](https://doi.org/10.48550/arXiv.2506.17857) (under review).

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![PyG](https://img.shields.io/badge/PyTorch-Geometric-orange.svg)](https://pyg.org/)
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![arXiv](https://img.shields.io/badge/arXiv-2506.17857-b31b1b.svg)](https://doi.org/10.48550/arXiv.2506.17857)
[![Conference](http://img.shields.io/badge/NeurIPS-2025-4b44ce.svg)](https://neurips.cc/)


## Requirements

To create a conda environment with all the dependencies, run the following commands:

```setup
# create a conda environment with the dependencies
conda update -f ./conda-env/waffle-gpu.yaml

# install waffle
pip install -e .
```

Copy and adjust the environment variables in the `.env` file.

```sh
cp .env.example .env
```

In .env you need to set the following variables:

```sh
ROOT_DIR=""  # the root directory of the project e.g. /workspace/AbRank-WALLE-Affinity
RUNS_PATH=""  # the path to the runs folder e.g. /workspace/AbRank-WALLE-Affinity/runs
DATA_PATH=""  # the path to the data folder e.g. /workspace/AbRank-WALLE-Affinity/data/local

# if you use wandb for logging
WANDB_PROJECT=""  # your wandb project name
WANDB_ENTITY=""  # your wandb entity
WANDB_API_KEY="" # your wandb api key obtained from wandb your profile page
```

NOTE: the environment variable `DATA_PATH` is used to set the `ROOT` path
argument of the dataset objects.

## Dataset

### Predicted Antibody and Antigen Structures

Predicted antibody and antigen structures are available on Kaggle accessible at
[AbRank: Antibody Affinity Ranking](https://www.kaggle.com/datasets/aurlienplissier/abrank).

Brief overview of the dataset:

![./figures/abrank-predicted-structures-kaggle.png](./figures/abrank-predicted-structures-kaggle.png)

### Pre-computed Embeddings

To train WAFFLE, you will need to download the pre-computed embeddings for
the antibodies and antigens to the folder
`./data/local/AbRank/processed/ca_10/pairdata`.

Pre-computed graph representations are accessible at 
[pairdata](https://drive.google.com/drive/folders/13BXfJW-hfbqx-wKrvei5sCuvXBV6m0kJ) (Total size: `146.57GB`), 
which contains 10 tarballs `00.tar.zst ... 09.tar.zst`. 
Download and decompress them to
`./data/local/AbRank/processed/ca_10/pairdata/`.

To decompress

```sh
# use 00.tar.zst as an example
tar -xf 00.tar.zst -C ./data/local/AbRank/processed/ca_10/pairdata/
```

### Splits

The processed split files will be automatically downloaded on the first run of the
`AbRankDataset` class.

Ranking splits:

```python
from waffle.data.components.abrank_dataset import AbRankDataset

dataset = AbRankDataset(root="./data/local")
```

Regression splits:

```python
from waffle.data.components.abrank_dataset_regression import AbRankDatasetRegression

dataset = AbRankDatasetRegression(root="./data/local")
```

Both will create a folder called `AbRank` under `./data/local/AbRank` and do
the following:

- `./data/local/AbRank/raw`: contain the downloaded tarballs
- `./data/local/AbRank/processed`: the pre-processed splits as csv files

## Training

To train the model(s), refer to the commands inlcuded in the `Makefile`.

We use `hydra` to manage the configuration files. To adjust hyperparameters, please refer to the `./waffle/config` folder.

For affinity RANKING tasks:

```train
# load the environment variables
source ./.env

# train the models under different settings
# 1. Balanced
train-abrank-ranking-balanced-train-swapped
# 2. Hard Ab
train-abrank-ranking-hard-ab-train-swapped
# 3. Hard Ag
train-abrank-ranking-hard-ag-train-swapped
```

For affinity REGRESSION tasks:

```train
# load the environment variables
source ./.env

# train the models under different settings
# 1. Balanced
train-abrank-regression-balanced-train-swapped
# 2. Hard Ab
train-abrank-regression-hard-ab-train-swapped
# 3. Hard Ag
train-abrank-regression-hard-ag-train-swapped
```

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Results

The following is the benchmarking results reported in the paper.

![./figures/benchmarking-results.png](./figures/benchmarking-results.png)

## Citation

> Under review, coming soon.
