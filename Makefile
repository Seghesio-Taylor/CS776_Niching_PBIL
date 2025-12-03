# Makefile for CS 776 Final Project
#
# This Makefile defines a set of convenience targets for running the
# various algorithms provided in this project.  Each target ensures
# that a Python virtual environment exists and the required Python
# packages are installed before invoking the corresponding script.
#
# Usage examples:
#   make env                 # create venv and install numpy/matplotlib
#   make ga_f2               # run GA on F2
#   make pbil_f3             # run vanilla PBIL on F3
#   make islands_base_f1     # Island PBIL speciation mode on F1
#   make run_all             # run the full sweep of experiments

# The Python interpreter and venv paths
PYTHON := python3
VENV   := .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

# Ensure venv exists and required packages are installed
ENV_OK := $(VENV)/bin/activate

all: help

help:
	@echo "Available targets:"
	@echo "  env                   - Create virtualenv and install numpy + matplotlib"
	@echo "  ga_f1 ga_f2 ga_f3     - Run GA on F1/F2/F3 with default settings"
	@echo "  pbil_f1 pbil_f2 pbil_f3 - Run vanilla PBIL on F1/F2/F3"
	@echo "  islands_none_f1/2/3   - Island PBIL, mode=none, 5 islands"
	@echo "  islands_base_f1/2/3   - Island PBIL, speciation (base), 5 islands"
	@echo "  islands_prop_f2/f3    - Island PBIL, proportional mode, 100 islands"
	@echo "  run_all               - Run full sweep of GA, PBIL and Island PBIL experiments"

# Create virtualenv and install required Python packages
env: $(ENV_OK)

$(ENV_OK):
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install numpy matplotlib
	@touch $(ENV_OK)

# ----------------- GA targets -----------------

ga_f1: env
	$(PY) GA.py --func F1 --runs 30

ga_f2: env
	$(PY) GA.py --func F2 --runs 30

ga_f3: env
	$(PY) GA.py --func F3 --runs 30

# ----------------- Vanilla PBIL targets -----------------

pbil_f1: env
	$(PY) vanilla_pbil.py --func F1 --runs 30

pbil_f2: env
	$(PY) vanilla_pbil.py --func F2 --runs 30

pbil_f3: env
	$(PY) vanilla_pbil.py --func F3 --runs 30

# ----------------- Island PBIL (mode=none, 5 islands) -----------------

islands_none_f1: env
	$(PY) pbil_island.py --func F1 --mode none --num_islands 5 --runs 30

islands_none_f2: env
	$(PY) pbil_island.py --func F2 --mode none --num_islands 5 --runs 30

islands_none_f3: env
	$(PY) pbil_island.py --func F3 --mode none --num_islands 5 --runs 30

# ----------------- Island PBIL speciation (base) -----------------

islands_base_f1: env
	$(PY) pbil_island.py --func F1 --mode base --num_islands 5 --runs 30

islands_base_f2: env
	$(PY) pbil_island.py --func F2 --mode base --num_islands 5 --runs 30

islands_base_f3: env
	$(PY) pbil_island.py --func F3 --mode base --num_islands 4 --runs 30

# ----------------- Island PBIL proportional (many islands) -----------------

islands_prop_f2: env
	$(PY) pbil_island.py --func F2 --mode proportional --num_islands 100 --runs 30

islands_prop_f3: env
	$(PY) pbil_island.py --func F3 --mode proportional --num_islands 100 --runs 30

# ----------------- Run all experiments -----------------

run_all: env
	bash run_all.sh