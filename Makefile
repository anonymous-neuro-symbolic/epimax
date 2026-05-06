# Project: Bipolar Possibilistic Framework
# Variables
PYTHON = python
CONFIG_DIR = configs
EXP_HIERARCHICAL = $(CONFIG_DIR)/exp1_hierarchical.yaml

.PHONY: help setup exp-hierarchical sweep clean

help:
	@echo "Available commands:"
	@echo "  make setup            - Install dependencies (Conda)"
	@echo "  make exp-hierarchical - Run the CIFAR-100 smoke test"
	@echo "  make clean            - Remove python cache and temp files"

setup:
	conda env create -f environment.yml
	@echo "Environment created. Run 'conda activate bipolar_env' to begin."

exp-hierarchical:
	$(PYTHON) main.py --config $(EXP_HIERARCHICAL)

sweep:
	$(PYTHON) sweep_runner.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache