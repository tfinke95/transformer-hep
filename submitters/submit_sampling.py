import os, json
import time
import numpy as np


def write_jobscript():
    with open(filename, "w") as f:
        f.write(
            f"""#!/usr/bin/env zsh

#SBATCH --account rwth0934
#SBATCH --job-name sample_{tag}

#SBATCH --output /home/bn227573/out/sample_{tag}_%J.log
#SBATCH --error /home/bn227573/out/sample_{tag}_%J_err.log

#SBATCH --time 45

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 2G

#SBATCH --gres=gpu:1

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

cd /home/bn227573/
conda activate torchProd
cd Projects/Transformers/physics_transformers

python sample_jets.py \\
    --model_dir {model_dir} \\
    --model_name {model_name} \\
    --savetag {savetag} \\
    --num_samples {num_samples} \\
    --batchsize {batchsize} \\
    --num_const {num_const} \\
    --seed {seed} \\
    --trunc {trunc}
"""
        )


tmp = os.listdir("/hpcwork/rwth0934/Transformers/")
MODEL_DIRS = [
    os.path.join("/hpcwork/rwth0934/Transformers/", x)
    for x in tmp
]
MODEL_NAMES = ["model_last.pt"] * len(MODEL_DIRS)
SAVETAGS = ["test_150"] * len(MODEL_DIRS)
NUM_SAMPLES = [100000] * len(MODEL_DIRS)
NUM_CONST = [150] * len(MODEL_DIRS)
SEEDS = [int(time.time())] * len(MODEL_DIRS)

n = 0
TRUNCS = [15000, 20000, 30000]
for trunc in TRUNCS:
    model_dir = "/hpcwork/rwth0934/Transformers/qcd_lowerq"
    model_name = "model_last.pt"
    savetags = [f"top{trunc}_train_100", f"top{trunc}_test_100"]
    num_samples = 100000
    batchsize = 100
    num_const = 100
    seed = SEEDS[0]
    tag = str(trunc)
    for savetag in savetags:
        if savetag == "test_100":
            seed = SEEDS[0] + 123678
        filename = f"jobscript_{trunc}_{savetag}.sh"
        write_jobscript()
        os.system(f"sbatch {filename}")
        # time.sleep(1)

# for i in range(len(MODEL_DIRS)):
#     if "many" in MODEL_DIRS[i]:
#         continue
#     if os.path.isfile(os.path.join(MODEL_DIRS[i], f"samples_{SAVETAGS[i]}.npz")):
#         continue
#     if not "lowerq" in MODEL_DIRS[i]:
#         continue
#     n += 1
#     filename = f"jobscript_{n}.sh"
#     print(MODEL_DIRS[i])
#     model_dir = MODEL_DIRS[i]
#     model_name = MODEL_NAMES[i]
#     savetag = SAVETAGS[i]
#     num_samples = NUM_SAMPLES[i]
#     batchsize = 100
#     num_const = NUM_CONST[i]
#     seed = SEEDS[i]
#     tag = model_dir.split("/")[-1]
#     write_jobscript()

#     # os.system(f"sbatch {filename}")
#     time.sleep(1)
