import os, json
import time

for bkg in ["qcd", "top"]:
    for tanh in [True, False]:
        tag = "_tanh_" if tanh else ""
        with open("jobscript.sh", "w") as f:
            f.write(
                f"""#!/usr/bin/env zsh
#SBATCH --account=rwth0934

#SBATCH --job-name 20negs{bkg}{tag}

#SBATCH --output /home/bn227573/out/negs_{bkg}{tag}_%J.log
#SBATCH --error /home/bn227573/out/negs_{bkg}{tag}_%J_err.log

#SBATCH --time 1350

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 2G

#SBATCH --gres=gpu:1

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

cd /home/bn227573/
conda activate torchEnv
cd Projects/AnomalyDetection/physics_transformers

python train_negatives.py \\
    --num_epochs 50 \\
    --data_path /hpcwork/bn227573/top_benchmark/train_{bkg}_30_bins.h5 \\
    --model_path models/20_fixed_new/20_fixed{tag}_{bkg}/model_last.pt \\
    --seed 0 \\
    --log_dir models/negatives_new/20_fixed_neg{tag}_{bkg} \\
    --batch_size 100 \\
    --num_events 600000 \\
    --num_const 20 \\
    --num_bins 41 31 31 \\
    --limit_const \\
    --logging_steps 100 \\
    --checkpoint_steps 5501 \\
    --lr 0.0001 \\
    --start_token \\
    {"--tanh" if tanh else ""}
"""
            )
        os.system("sbatch jobscript.sh")
