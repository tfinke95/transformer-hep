import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


def preprocess_dataframe(
    df,
    num_features,
    num_bins,
    num_const,
    to_tensor=True,
    reverse=False,
    start=False,
    end=False,
    limit_nconst=False,
):
    x = df.to_numpy(dtype=np.int64)[:, : num_const * num_features]
    x = x.reshape(x.shape[0], -1, num_features)
    padding_mask = x[:, :, 0] != -1

    if limit_nconst:
        keepings = padding_mask.sum(-1) >= num_const
        x = x[keepings]
        padding_mask = padding_mask[keepings]

    if reverse:
        print("Reversing pt order")
        x[x == -1] = np.max(num_bins) + 10
        idx_sort = np.argsort(x[:, :, 0], axis=-1)
        for i in range(len(x)):
            x[i] = x[i, idx_sort[i]]
        x[x == np.max(num_bins) + 10] = -1

    num_prior_bins = np.cumprod((1,) + num_bins[:-1])
    bins = (x * num_prior_bins.reshape(1, 1, num_features)).sum(axis=2)

    if start:
        print("Adding start particles")
        bins = np.concatenate(
            (np.ones((len(bins), 1), dtype=int) * -100, bins),
            axis=1,
        )

        x = np.concatenate(
            (
                np.zeros((len(x), 1, num_features), dtype=int),
                x,
            ),
            axis=1,
        )
        padding_mask = x[:, :, 0] != -1
        bins[~padding_mask] = -100
    else:
        bins[~padding_mask] = -100

    if end:
        print("Adding stop token")
        seq_lengths = padding_mask.sum(-1)
        x = np.append(x, -np.ones((x.shape[0], 1, x.shape[2]), dtype=int), axis=1)
        x[np.arange(x.shape[0]), seq_lengths] = 0
        bins = np.append(bins, -100 * np.ones((bins.shape[0], 1)).astype(int), axis=1)
        bins[np.arange(bins.shape[0]), seq_lengths] = np.prod(num_bins)
        padding_mask = x[:, :, 0] != -1

    if to_tensor:
        x = torch.tensor(x)
        padding_mask = torch.tensor(padding_mask)
        bins = torch.tensor(bins)
    return x, padding_mask, bins


def imagePreprocessing(jets, filename=None):
    def center():
        mean_eta = np.average(constituents[:, 1], weights=constituents[:, 0])
        mean_phi = np.average(constituents[:, 2], weights=constituents[:, 0])

        constituents[:, 2] -= mean_phi
        constituents[:, 1] -= mean_eta

    def rotate():
        # Calculate the major axis
        eta_coords = (constituents[:, 1] - 15) * constituents[:, 0]
        phi_coords = (constituents[:, 2] - 15) * constituents[:, 0]
        coords = np.vstack([eta_coords, phi_coords])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)
        sorted_indices = np.argsort(evals)[::-1]
        major_axis = evecs[:, sorted_indices[0]]

        # Rotate major axis to have 0 phi
        theta = np.arctan(major_axis[0] / major_axis[1])
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, s], [-s, c]])
        constituents[:, 1:3] = np.matmul(constituents[:, 1:3], rotation)

    def flip():
        quad1 = 0
        quad2 = 0
        quad3 = 0
        quad4 = 0

        for i in range(len(constituents)):
            if constituents[i, 1] > 0:
                if constituents[i, 2] > 0:
                    quad1 += constituents[i, 0]
                else:
                    quad2 += constituents[i, 0]
            else:
                if constituents[i, 2] > 0:
                    quad3 += constituents[i, 0]
                else:
                    quad4 += constituents[i, 0]

        quad = np.argmax([quad1, quad2, quad3, quad4])

        if quad == 1:
            constituents[:, 2] *= -1
        elif quad == 2:
            constituents[:, 1] *= -1
        elif quad == 3:
            constituents[:, 1] *= -1
            constituents[:, 2] *= -1

    print("Started advancedPreProcess")
    # Loop over all jets
    for i in tqdm(range(np.shape(jets)[0])):
        constituents = jets[i]

        center()
        rotate()
        flip()

        # Normalise pT of the jet to 1
        constituents[:, 0] /= np.sum(constituents[:, 0])
        jets[i] = constituents

    print(f"Exiting advancedPreProcess, shape: {np.shape(jets)}")

    return jets


if __name__ == "__main__":
    df = pd.read_hdf(
        "/mnt/wsl/PHYSICALDRIVE1p1/thorben/Data/jet_datasets/top_benchmark/v0/train_qcd_30_bins.h5",
        key="discretized",
        stop=10000,
    )
    x, pad, bin = preprocess_dataframe(
        df=df,
        num_features=3,
        num_bins=(41, 31, 31),
        num_const=30,
        to_tensor=False,
        reverse=False,
        start=True,
        end=True,
        limit_nconst=False,
    )
    print(x.shape, pad.shape, bin.shape)
    print(x[0])
    print(bin[0])
