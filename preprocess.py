import pandas as pd
import numpy as np
import torch


def preprocess_dataframe(df, num_features, num_bins, num_const, num_events,
                         to_tensor=True):
    x = df.to_numpy(dtype=np.int64)[:num_events, :num_const*num_features]
    x = x.reshape(x.shape[0], -1, num_features)
    padding_mask = x[:, :, 0] != -1

    num_prior_bins = np.cumprod((1,) + num_bins[:-1])
    bins = (x * num_prior_bins.reshape(1, 1, num_features)).sum(axis=2)
    bins[~padding_mask] = -100

    if to_tensor:
        x = torch.tensor(x)
        padding_mask = torch.tensor(padding_mask)
        bins = torch.tensor(bins)
    return x, padding_mask, bins
