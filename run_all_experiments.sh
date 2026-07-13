#!/bin/bash
source .venv/bin/activate
# python experiments/run_experiment.py glove --n_queries=500 --start_blocks=10 --end_blocks=201 --step=10
# python experiments/run_experiment.py mf_dino2 --n_queries=500 --start_blocks=10 --end_blocks=201 --step=10
# python experiments/run_experiment.py laion --n_queries=500 --start_blocks=10 --end_blocks=201 --step=10
python experiments/run_experiment.py gooaq --n_queries=500 --start_blocks=10 --end_blocks=201 --step=10
