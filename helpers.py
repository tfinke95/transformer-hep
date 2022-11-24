import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from preprocess import imagePreprocessing, preprocess_dataframe
import pandas as pd


def get_samples(model, data_loader, device):
    if type(model) is str:
        model = torch.load(model)
    model.to(device)
    model.classifier = False

    samples = []
    bins = []
    i = 0
    for x, _, _ in tqdm(iterable=data_loader, desc="Batch", total=len(data_loader)):
        _samples, _sample_bins = model.sample(x[:, 0].to(device), device)
        i += 1
        samples.append(_samples.cpu())
        bins.append(_sample_bins.cpu())
    samples = torch.concat(samples, 0).cpu().numpy()
    bins = torch.concat(bins, 0).cpu().numpy()
    return samples, bins


def jets_to_images(data):
    jets = imagePreprocessing(data.astype(float))
    print(jets.shape)
    images = np.zeros((len(jets), 30, 30))
    for i in tqdm(range(len(jets))):
        images[i], _, _ = np.histogram2d(
            jets[i, :, 1],
            jets[i, :, 2],
            bins=(np.arange(-15.5, 15.5, 1), np.arange(-15.5, 15.5, 1)),
            weights=jets[i, :, 0],
        )
    return images


def get_data(
    N: int,
    files: list[str],
    tags: list[str],
    reverse=False,
):
    assert len(files) == len(
        tags
    ), f"Need same number of tags and files (given {len(tags)} {len(files)}"
    data_loaders = {}
    orig_data = {}
    for ind, file in enumerate(files):
        df = pd.read_hdf(
            file,
            key="discretized",
            stop=N,
        )
        jets, mask, bins = preprocess_dataframe(
            df,
            num_features=3,
            num_bins=(41, 31, 31),
            num_const=20,
            limit_nconst=True,
            reverse=reverse,
        )
        loader = DataLoader(
            TensorDataset(jets, mask, bins), batch_size=100, shuffle=False
        )
        data_loaders[tags[ind]] = loader
        orig_data[tags[ind]] = (jets, mask, bins)

    return data_loaders, orig_data


if __name__ == "__main__":
    pass
