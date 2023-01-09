import os, json
import time
import numpy as np

NUM_LAYERS = [
    6,
]
HIDDEN_DIMS = [
    128,
]
HEADS = [
    4,
]
DROPOUTS = [
    0.1,
]
LRS = [
    1e-3,
    1e-4,
    5e-4,
    5e-5,
]
# with open("scan_params.txt", "w") as f:
#     f.write(f"{'Layers':8s}{'Hidden':8s}{'Heads':8s}{'Dropout':8s}\n")

# for i in NUM_LAYERS:
#     for j in HIDDEN_DIMS:
for lr in LRS:
    num_layer = 6  # np.random.choice(NUM_LAYERS)
    hidden_dim = 128  # np.random.choice(HIDDEN_DIMS)
    heads = 4  # np.random.choice(HEADS)
    dropout = 0.1  # np.random.choice(DROPOUTS)

    with open("scan_params.txt", "a") as f:
        f.write(
            f"{str(num_layer):^8s}{str(hidden_dim):^8s}{str(heads):^8s}{str(dropout):^8s}\n"
        )

    with open("jobscript.sh", "w") as f:
        f.write(
            f"""#!/usr/bin/env zsh

#SBATCH --job-name LR_{lr}

#SBATCH --output /home/bn227573/out/LR_{lr}_%J.log
#SBATCH --error /home/bn227573/out/LR_{lr}_%J_err.log

#SBATCH --time 15

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 2G

#SBATCH --gres=gpu:1

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

cd /home/bn227573/
conda activate torchEnv
cd Projects/AnomalyDetection/physics_transformers

python train.py \\
    --num_epochs 50 \\
    --data_path /hpcwork/bn227573/top_benchmark/train_qcd_30_bins.h5 \\
    --seed 0 \\
    --log_dir /hpcwork/bn227573/Transformers/models/lr_run/lr_{lr} \\
    --batch_size 100 \\
    --num_events 600000 \\
    --num_const 50 \\
    --num_bins 41 31 31 \\
    --logging_steps 50 \\
    --checkpoint_steps 0 \\
    --lr {lr} \\
    --num_layers {num_layer} \\
    --hidden_dim {hidden_dim} \\
    --num_heads {heads} \\
    --dropout {dropout} \\
    --start_token \\
    --end_token \\
    --tanh
"""
        )
    os.system("sbatch jobscript.sh")
    time.sleep(1)
