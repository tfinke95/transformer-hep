import os, json
import time


job_n = 0

config = json.load(open("config_default.json", "r"))

config["data"]["bg_file"] = "/hpcwork/bn227573/top_benchmark/test_qcd_30_bins.h5"
config["data"]["bg_key"] = "raw"
config["data"]["sig_key"] = "discretized"
config["data"]["f1"] = [0.0]
config["data"]["f2"] = [1.0]
config["data"]["n_const"] = 50
config["data"]["n_jets"] = 100000
config["data"]["seed"] = int(time.time())
config["data"]["sig_files"] = [
    # "/hpcwork/bn227573/Transformers/models/end_token/end_token_noAdd_qcd/samples_50.npz",
    "/hpcwork/bn227573/top_benchmark/test_qcd_30_bins.h5",
]
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


for noise in [True, False]:
    bg_settings = ["raw"] if noise else ["noise", "raw"]
    for bg_setting in bg_settings:
        job_n += 1
        filename = f"jobscript_{job_n}.sh"

        config["data"]["sig_noise"] = noise
        if bg_setting == "noise":
            config["data"]["bg_key"] = "discretized"
            config["data"]["bg_noise"] = True
        elif bg_setting == "raw":
            config["data"]["bg_key"] = "raw"
            config["data"]["bg_noise"] = None

        config["logging"][
            "logfolder"
        ] = f"logs/data_qcd_{bg_setting}_vs_{'noise' if noise else 'discrete'}"
        config_file = f"config_qcd_{job_n}.json"

        json.dump(config, open(config_file, "w"), sort_keys=True, indent=2)

        with open(filename, "w") as f:
            f.write(
                f"""#!/usr/bin/env zsh

#SBATCH --job-name frac_{noise}_{job_n}
#SBATCH --account rwth0934

#SBATCH --output /home/bn227573/out/pnet_raw_{noise}_%J.log
#SBATCH --error /home/bn227573/out/pnet_raw_{noise}_%J_err.log

#SBATCH --time 130

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 1G

#SBATCH --gres=gpu:1


cd /home/bn227573
source .zshrc
conda activate monoJet

cd Projects/AnomalyDetection/physics_transformers/particlenet

python train.py {config_file}
"""
            )
        os.system("sbatch {}".format(filename))
        time.sleep(1)
        print(f"Submitted {filename}")
