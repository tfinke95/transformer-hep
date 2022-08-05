import pandas as pd
import numpy as np
import torch


def preprocess_dataframe(df, num_features, num_bins, num_const, num_events,
                         to_tensor=True, reverse=False, start_end=True):
    x = df.to_numpy(dtype=np.int64)[:num_events, :num_const*num_features]
    x = x.reshape(x.shape[0], -1, num_features)

    if reverse:
        print('Reverting pt order')
        x[x==-1] = np.max(num_bins) + 10
        idx_sort = np.argsort(x[:, :, 0], axis=-1)
        for i in range(len(x)):
            x[i] = x[i, idx_sort[i]]
        x[x==np.max(num_bins)+10] = -1

    padding_mask = x[:, :, 0] != -1

    num_prior_bins = np.cumprod((1,) + num_bins[:-1])
    bins = (x * num_prior_bins.reshape(1, 1, num_features)).sum(axis=2)

    if start_end:
        bins[~padding_mask] = 39401
        bins = np.append(bins, np.ones((len(bins), 1), dtype=int) * 39401, axis=1,)
        bins = np.concatenate((np.ones((len(bins), 1), dtype=int) * 39402, bins), axis=1)

        x = np.concatenate((np.zeros((len(x), 1, num_features), dtype=int),
                            x,
                            -np.ones((len(x), 1, num_features), dtype=int)),
                            axis=1)
    else:
        bins[~padding_mask] = -100

    padding_mask = x[:, :, 0] != -1

    if to_tensor:
        x = torch.tensor(x)
        padding_mask = torch.tensor(padding_mask)
        bins = torch.tensor(bins)
    return x, padding_mask, bins
