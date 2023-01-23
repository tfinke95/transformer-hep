import numpy as np
import sys, json, os
import pandas as pd


def transform_momenta(momenta, features="last"):
    pts = momenta[:, :, 0]
    mask = pts != -100
    etas = momenta[:, :, 1]
    phis = momenta[:, :, 2]

    drs = np.sqrt(np.sum(np.square(momenta[:, :, 1:3]), -1))

    pxs = np.cos(phis) * pts * mask
    pys = np.sin(phis) * pts * mask
    pxj = np.sum(pxs, -1)
    pyj = np.sum(pys, 1)

    ptj = np.sqrt(pxj**2 + pyj**2)

    newVec = np.stack(
        [etas, phis, np.log(pts), np.log(pts / ptj.reshape(-1, 1)), drs],
        -1,
    )
    for i in range(newVec.shape[-1]):
        newVec[~mask, i] = -100
    return newVec


def get_config(test=False):
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        config = json.load(open(sys.argv[1], "r"))
    else:
        assert not test, f"Provide config file used for training"
        config = json.load(open("config.json", "r"))

    if not test:
        i = 0
        original = config["logging"]["logfolder"]
        while os.path.isdir(config["logging"]["logfolder"]):
            i += 1
            config["logging"]["logfolder"] = f"{original}_{i}"

        os.makedirs(config["logging"]["logfolder"])
        json.dump(
            config,
            open(os.path.join(config["logging"]["logfolder"], "config.json"), "w"),
            sort_keys=True,
            indent=2,
        )

    return config


def make_continues(jets, noise=False):
    pt_bins = np.load("../preprocessing_bins/pt_bins_30_bins.npy")
    eta_bins = np.load("../preprocessing_bins/eta_bins_30_bins.npy")
    phi_bins = np.load("../preprocessing_bins/phi_bins_30_bins.npy")

    pt_disc = jets[:, :, 0]
    print(pt_disc.max(), pt_bins.shape)
    mask = pt_disc == 0
    eta_disc = jets[:, :, 1]
    print(eta_disc.max(), eta_bins.shape)
    phi_disc = jets[:, :, 2]
    print(phi_disc.max(), phi_bins.shape)

    if noise:
        pt_con = (pt_disc - np.random.uniform(0.0, 1.0, size=pt_disc.shape)) * (
            pt_bins[1] - pt_bins[0]
        ) + pt_bins[0]
        eta_con = (eta_disc - np.random.uniform(0.0, 1.0, size=eta_disc.shape)) * (
            eta_bins[1] - eta_bins[0]
        ) + eta_bins[0]
        phi_con = (phi_disc - np.random.uniform(0.0, 1.0, size=phi_disc.shape)) * (
            phi_bins[1] - phi_bins[0]
        ) + phi_bins[0]
    else:
        pt_con = pt_disc * (pt_bins[1] - pt_bins[0]) + pt_bins[0]
        eta_con = eta_disc * (eta_bins[1] - eta_bins[0]) + eta_bins[0]
        phi_con = phi_disc * (phi_bins[1] - phi_bins[0]) + phi_bins[0]

    continues_jets = np.stack((np.exp(pt_con), eta_con, phi_con), -1)
    continues_jets[mask] = -100

    return continues_jets


def load_data(params, test=False, plot_dists=None):
    def load_file(file, key=None):
        # Load  sample file
        if file.endswith("npz"):
            if test:
                file = file.replace("50", "100_test")
            print(file)
            dat = np.load(file)["jets"][: params["n_jets"], : params["n_const"]]
        # Load data file
        elif file.endswith("h5"):
            if test:
                dat = pd.read_hdf(
                    file,
                    key=key,
                    start=100000,
                    stop=100000 + params["n_jets"],
                )
            else:
                dat = pd.read_hdf(file, key=key, stop=params["n_jets"])
            dat = dat.to_numpy()[:, : params["n_const"] * 3]
            dat = dat.reshape(dat.shape[0], -1, 3)
        else:
            assert False, "Filetype for bg not supported"
        # Delete empty jest (can occur in samplings)
        dat = np.delete(dat, np.where(dat[:, 0, 0] == 0)[0], axis=0)
        dat[dat == -1] = 0
        return dat

    print(params["bg_file"])
    print(params["sig_files"])

    bg = load_file(params["bg_file"], key=params["bg_key"])
    if params["bg_key"] == "discretized":
        print("BG made continuous")
        bg = make_continues(bg, params["bg_noise"])
    else:
        bg[bg[:, :, 0] == 0] = -100

    sig = load_file(params["sig_files"][0], key=params["sig_key"])
    if params["sig_key"] == "discretized":
        print("Sig made continuous")
        sig = make_continues(sig, params["sig_noise"])
    else:
        sig[sig[:, :, 0] == 0] = -100
    print(bg.shape, sig.shape)

    data = np.append(bg, sig, 0)
    labels = np.append(np.zeros(len(bg)), np.ones(len(sig)))
    shuffle = np.random.permutation(len(data))

    data = transform_momenta(data)

    if not plot_dists is None:
        import matplotlib.pyplot as plt

        features = data.shape[-1]
        fig, axes = plt.subplots(
            features, 1, constrained_layout=True, figsize=(features * 3, 10)
        )
        for i in range(data.shape[-1]):
            range_min = data[:, :, i][data[:, :, i] != -100].min()
            range_max = data[:, :, i].max()
            axes[i].hist(
                data[labels == 0, :, i][data[labels == 0, :, 0] != -100].flatten(),
                bins=300,
                range=[range_min, range_max],
                histtype="step",
                density=True,
                label="Background",
            )
            axes[i].hist(
                data[labels == 1, :, i][data[labels == 1, :, 0] != -100].flatten(),
                bins=300,
                range=[range_min, range_max],
                histtype="step",
                density=True,
                label="Signal",
            )
        axes[0].legend()
        fig.savefig(plot_dists)
        plt.close(fig)

    return data[shuffle], labels[shuffle]


if __name__ == "__main__":
    pass
