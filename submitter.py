import os, json
import time

for bkg in ['top', 'qcd']:
    for reverse in [True, False,]:
        with open('jobscript.sh', 'w') as f:
            f.write(f"""#!/usr/bin/env zsh
#SBATCH --account=rwth0934

#SBATCH --job-name 20{bkg}{'R' if reverse else ''}

#SBATCH --output /home/bn227573/out/20{bkg}{'R' if reverse else ''}_%J.log
#SBATCH --error /home/bn227573/out/20{bkg}{'R' if reverse else ''}_%J_err.log

#SBATCH --time 400

#SBATCH --mem-per-cpu 8G

#SBATCH --gres=gpu:1

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

cd /home/bn227573/
conda activate torchEnv
cd Projects/AnomalyDetection/physics_transformers

python train.py \\
    --num_epochs 100 \\
    --data_path /hpcwork/bn227573/top_benchmark/train_{bkg}_30_bins.h5 \\
    --model_dir models/20_fixed/20_fixed{'_reverse' if reverse else ''}_{bkg} \\
    --batch_size 100 \\
    --num_events 600000 \\
    --num_const 20 \\
    --num_bins 41 31 31 \\
    --limit_const \\
    {'--reverse' if reverse else ''}
""")
        os.system("sbatch jobscript.sh")
