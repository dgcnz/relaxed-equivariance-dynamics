# DL2

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

Print slurm logs given job id

```sh
make slurmtrain experiment=
```

```sh
make slurmcat id=6246500
```