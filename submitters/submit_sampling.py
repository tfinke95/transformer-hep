import os, json
import time
import numpy as np


def write_jobscript():
    with open("jobscript.sh", "w") as f:
        f.write(
            f"""#!/usr/bin/env zsh

#SBATCH --job-name sample_{tag}

#SBATCH --output /home/bn227573/out/sample_{tag}_%J.log
#SBATCH --error /home/bn227573/out/sample_{tag}_%J_err.log

#SBATCH --time 30

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 2G

#SBATCH --gres=gpu:1

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

cd /home/bn227573/
conda activate torchEnv
cd Projects/AnomalyDetection/physics_transformers

python sample_jets.py \\
    --model_dir {model_dir} \\
    --model_name {model_name} \\
    --savetag {savetag} \\
    --num_samples {num_samples} \\
    --batchsize {batchsize} \\
    --num_const {num_const} \\
    --seed {seed}
"""
        )


tmp = os.listdir("/hpcwork/bn227573/Transformers/models/end_token/")
MODEL_DIRS = [
    os.path.join("/hpcwork/bn227573/Transformers/models/end_token/", x)
    for x in tmp
    if "noAdd" in x
]
MODEL_NAMES = ["model_last.pt"] * len(MODEL_DIRS)
SAVETAGS = ["100_test"] * len(MODEL_DIRS)
NUM_SAMPLES = [20000] * len(MODEL_DIRS)
NUM_CONST = [100] * len(MODEL_DIRS)
SEEDS = [19950107] * len(MODEL_DIRS)


for i in range(len(MODEL_DIRS)):
    model_dir = MODEL_DIRS[i]
    model_name = MODEL_NAMES[i]
    savetag = SAVETAGS[i]
    num_samples = NUM_SAMPLES[i]
    batchsize = 100
    num_const = NUM_CONST[i]
    seed = SEEDS[i]
    tag = model_dir.split("/")[-1]
    write_jobscript()

    os.system("sbatch jobscript.sh")
    time.sleep(1)
