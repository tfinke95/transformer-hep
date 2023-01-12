import numpy as np
import sys, json, os
import pandas as pd


def transform_momenta(momenta, features="last"):
    pts = momenta[:, :, 0]
    mask = pts == 0
    etas = momenta[:, :, 1]
    phis = momenta[:, :, 2]

    drs = np.sqrt(np.sum(np.square(momenta[:, :, 1:3]), -1))

    pxs = np.cos(phis) * pts
    pys = np.sin(phis) * pts
    pxj = np.sum(pxs, -1)
    pyj = np.sum(pys, 1)

    ptj = np.sqrt(pxj**2 + pyj**2)

    newVec = np.stack(
        [etas, phis, np.log(pts), np.log(pts / ptj.reshape(-1, 1)), drs],
        -1,
    )
    for i in range(newVec.shape[-1]):
        newVec[mask, i] = 0
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
    mask = pt_disc == 0
    eta_disc = jets[:, :, 1]
    phi_disc = jets[:, :, 2]

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
    continues_jets[mask] = 0

    return continues_jets


def load_data(params):
    def load_file(file):
        if file.endswith("npz"):
            dat = np.load(file)["jets"][: params["n_jets"], : params["n_const"]]
        elif file.endswith("h5"):
            dat = pd.read_hdf(file, key="discretized", stop=params["n_jets"])
            dat = dat.to_numpy(dtype=np.int64)[:, : params["n_const"] * 3]
            dat = dat.reshape(dat.shape[0], -1, 3)
        else:
            assert False, "Filetype for bg not supported"
        dat = np.delete(dat, np.where(dat[:, 0, 0] == 0)[0], axis=0)
        dat[dat == -1] = 0
        return dat

    bg = load_file(params["bg_file"])
    sig = load_file(params["sig_files"][0])

    data = np.append(bg, sig, 0)
    labels = np.append(np.zeros(len(bg)), np.ones(len(sig)))
    shuffle = np.random.permutation(len(data))
    data = make_continues(data, params["cont_noise"])
    data = transform_momenta(data)

    # import matplotlib.pyplot as plt

    # features = data.shape[-1]
    # fig, axes = plt.subplots(
    #     features, 1, constrained_layout=True, figsize=(features * 3, 10)
    # )
    # for i in range(data.shape[-1]):
    #     axes[i].hist(
    #         data[labels == 0, :, i][data[labels == 0, :, 2] != 0].flatten(),
    #         bins=100,
    #         histtype="step",
    #     )
    #     axes[i].hist(
    #         data[labels == 1, :, i][data[labels == 1, :, 2] != 0].flatten(),
    #         bins=100,
    #         histtype="step",
    #     )
    # plt.show()

    return data[shuffle], labels[shuffle]


if __name__ == "__main__":
    pass
