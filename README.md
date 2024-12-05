# Relaxed equivariance dynamics

This repository contains the official implementation of our blogpost titled: **Effect of equivariance on training dynamics**.

Our blogpost has been published at [GRaM Workshop @ ICML 2024](https://gram-blogposts.github.io/blog/2024/relaxed-equivariance/).

Our extended blogpost can be found [here](blogpost.md).

To cite our work:

```
@misc{equivdyna_2024,
author = {Canez, Diego and Midavaine, Nesta and Stessen, Thijs and Fan, Jiapeng and Arias, Sebastian},
doi = {10.5281/zenodo.14283183},
journal = {GRaM Workshop, ICML 2024},
month = jul,
title = {{Effect of equivariance on training dynamics}},
year = {2024}
}
```

## Package contributions

To maximize convenience, reproducibility and encourage usage of our modules (models, datasets, tools), we've package some of them separately.


- [gconv: (Relaxed) Regular Group Convolutions package](https://github.com/dgcnz/gconv)
- [JHTDB HuggingFace Dataset](https://huggingface.co/datasets/dl2-g32/jhtdb)

## Setup 

Make virtual environment and install dependencies:
```sh
make setup_env
```

Source your virtual environment:
```sh
source .venv/bin/activate
```

## Local Usage

Reproduce the training results from a given experiment:

```sh
python -m src.train experiment=wang2024/rgcnn_triple
```

## SLURM Usage

### Pre-requisites
- ssh into snellius
- Move into the project root
    ```sh
    cd ~/development/dl2
    ```

### Train model based on experiment config

Let's say you want to run the experiment at `configs/experiment/wang2022/equivariance_test/convnet.yaml`. You can make use of the shortcut `slurmtrain` as follows:

```sh
make strain experiment=wang2022/equivariance_test/convnet
```

If you need to modify anything, the script is at `scripts/slurm/train.sh`.

### Print slurm logs given job id

```sh
make slurmcat id=6246500
```

The logs are stored at `scripts/slurm_logs/slurm_output_{id}.out`.
