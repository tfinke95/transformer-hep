import os, json
import time
import numpy as np

data_files = {}
data_files["qcd"] = "/hpcwork/bn227573/top_benchmark/test_qcd_30_bins.h5"
data_files["top"] = "/hpcwork/bn227573/top_benchmark/test_top_30_bins.h5"

sample_files = []
params = []
for t in ["_tanh", ""]:
    for s in ["qcd", "top"]:
        for c in [50, 100]:
            sample_files.append(
                f"/hpcwork/bn227573/Transformers/models/end_token/end_token_noAdd{t}_{s}/samples_{c}.npz"
            )
            params.append([c, data_files[s], f"sample_tests/{s}{t}_{c}", f"{s+t}_{c}"])

for x, y in zip(sample_files, params):

    with open("jobscript.sh", "w") as f:
        f.write(
            f"""#!/usr/bin/env zsh

#SBATCH --job-name {y[-1]}

#SBATCH --output /home/bn227573/out/{y[-1]}_%J.log
#SBATCH --error /home/bn227573/out/{y[-1]}_%J_err.log

#SBATCH --time 20

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 2G

#SBATCH --gres=gpu:1

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

cd /home/bn227573/
conda activate torchEnv
cd Projects/AnomalyDetection/physics_transformers

python test_samples.py --bg {y[1]} --sig {x} -N 100000 -c {y[0]} -E 50 --save_dir {y[2]}
"""
        )
    os.system("sbatch jobscript.sh")
    print(x, y)
    time.sleep(0.5)
