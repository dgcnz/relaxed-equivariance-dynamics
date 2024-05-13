VENV           = .venv
VENV_PYTHON    = $(VENV)/bin/python
SYSTEM_PYTHON  = $(or $(shell which python3), $(shell which python))
PYTHON = $(VENV_PYTHON)

help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean: ## Clean autogenerated files
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

clean-logs: ## Clean logs
	rm -rf logs/**

format: ## Run pre-commit hooks
	pre-commit run -a

sync: ## Merge changes from main branch to your current branch
	git pull
	git pull origin main

test: ## Run not slow tests
	pytest -k "not slow"

test-full: ## Run all tests
	pytest

train: ## Train the model
	python src/train.py


send_key: ## Sends public key to snellius
	ssh -i ~/.ssh/surf dl2


module_avail: ## greppable module avail
	module -t avail 2>&1

load_modules:
	module load 2023
	module load Python/3.11.3-GCCcore-12.3.0

unload_modlues:
	module unload Python/3.11.3-GCCcore-12.3.0
	module unload 2023

setup_env:
	$(SYSTEM_PYTHON) -m venv .venv
	$(PYTHON) -m pip install poetry
	$(PYTHON) -m poetry install


