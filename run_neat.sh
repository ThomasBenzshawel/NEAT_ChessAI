#!/bin/bash

#SBATCH --job-name="NEAT Training"
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=benzshawelt@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:2
#SBATCH --cpus-per-gpu=8
#SBATCH --account=practicum


SCRIPT_NAME="Neat Training"
CONTAINER="/data/containers/msoe-tensorflow-23.05-tf2-py3.sif"


PYTHON_FILE="run_problem_sim.py"
SCRIPT_ARGS=""


## SCRIPT
echo "SBATCH SCRIPT: ${SCRIPT_NAME}"

# Pip install the requirements
pip install chess
pip install abc


srun hostname; pwd; date;
srun singularity exec --nv -B /data:/data ${CONTAINER} python3 ${PYTHON_FILE} ${SCRIPT_ARGS}

echo "END: " $SCRIPT_NAME