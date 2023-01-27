import os, json
import time

submit = True

tag = "qcdVStop_full"

def write_jobscript():
    with open(filename, "w") as f:
        f.write(
            f"""#!/usr/bin/env zsh

#SBATCH --job-name pNet_{tag}_{setting}_{job_n}
#SBATCH --account rwth0934

#SBATCH --output /home/bn227573/out/pnet_{tag}_{setting}_%J.log
#SBATCH --error /home/bn227573/out/pnet_{tag}_{setting}_%J_err.log

#SBATCH --time 600

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 3G

#SBATCH --gres=gpu:1


cd /home/bn227573
source .zshrc
conda activate monoJet

cd Projects/Transformers/physics_transformers/particlenet

python train.py {config_file}
"""
        )

job_n = 0

config = json.load(open("config_default.json", "r"))

config["data"]["bg_file"] = "/hpcwork/rwth0934/top_benchmark/discretized/train_qcd_pt40_eta30_phi30_lower001.h5"
config["data"]["bg_key"] = "discretized"
config["data"]["sig_file"] = "/hpcwork/rwth0934/top_benchmark/discretized/train_top_pt40_eta30_phi30_lower001.h5"
config["data"]["sig_key"] = "discretized"
config["data"]["n_const"] = 50
config["data"]["n_jets"] = 600000
config["data"]["seed"] = int(time.time())
config["data"]["bg_noise"] = False
config["data"]["sig_noise"] = False

config["graphnet"]["activation"] = "LeakyReLU"
config["graphnet"]["channels"] = [[64, 64, 64], [128, 128, 128], [256, 256, 256]]
config["graphnet"]["classifier"] = [256, 128, 2]
config["graphnet"]["dropout"] = 0.1
config["graphnet"]["k"] = 16
config["graphnet"]["static"] = False

config["logging"]["logfolder"] = "logs/discrete"

config["mask"] = True

config["training"]["batch_size"] = 256
config["training"]["epochs"] = 75
config["training"]["validation_split"] = 0.1
config["training"]["validation_freq"] = 1
config["training"]["verbose"] = 2



# Run top vs QCD as data test for discrete and continuous data
for setting in ["discretized"]:
    job_n += 1
    filename = f"jobscripts/jobscript_data_{job_n}.sh"

    config["data"]["bg_key"] = setting
    config["data"]["sig_key"] = setting

    config["logging"][
        "logfolder"
    ] = f"logs/data_qcdVStop_{setting}_full"
    config_file = f"configs/config_data_{job_n}.json"

    json.dump(config, open(config_file, "w"), sort_keys=True, indent=2)
    write_jobscript()

    if submit:
        os.system("sbatch {}".format(filename))
    print(f"Submitted {filename}")

# Run for samples
bg_files = [
    "/hpcwork/rwth0934/top_benchmark/discretized/val_qcd_pt40_eta30_phi30.h5",
    "/hpcwork/rwth0934/top_benchmark/discretized/val_top_pt40_eta30_phi30.h5",
]
sig_files = [
    "/hpcwork/rwth0934/Transformers/qcd_50/samples_train_50.npz",
    "/hpcwork/rwth0934/Transformers/top_50/samples_train_50.npz",
]
tag = "samples"

job_n = 0
for bg, sig in zip(bg_files, sig_files):
    job_n += 1
    filename = f"jobscripts/jobscript_samples_{job_n}.sh"
    config["logging"]["logfolder"] = f"logs/samples_{'qcd' if job_n==1 else 'top'}_mask0"
    config["data"]["bg_file"] = bg
    config["data"]["bg_key"] = "discretized"
    config["data"]["sig_file"] = sig
    config["data"]["sig_key"] = "discretized"
    config_file = f"configs/config_samples_{job_n}.json"
    json.dump(config, open(config_file, "w"), sort_keys=True, indent=2)
    write_jobscript()
    if submit:
        os.system("sbatch {}".format(filename))
    print(f"Submitted {filename}")
