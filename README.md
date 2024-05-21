# DL2
Our blogpost can be found [here](blogpost.md).

## Package contributions

To maximize convenience, reproducibility and encourage usage of our modules (models, datasets, tools), we've package some of them separately.


- [gconv: (Relaxed) Regular Group Convolutions package](https://github.com/dgcnz/gconv)
- [neuralyze: Equivariance analysis package](https://github.com/dgcnz/neuralyze)
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
