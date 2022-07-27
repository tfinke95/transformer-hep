import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model import JetTransformer
from preprocess import preprocess_dataframe

from argparse import ArgumentParser
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve

torch.multiprocessing.set_sharing_strategy('file_system')

def load_model(model_dir, name):
    model = torch.load(os.path.join(model_dir, f'model_{name}.pt'))
    return model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='models/test', help="Model directory")
    parser.add_argument("--model_name", type=str, default='last', help="Checkpoint name")
    parser.add_argument("--data_path", type=str, default='Datasets/', help="Path to training data file")
    parser.add_argument("--data_split", type=str, default='test', help="Split to evaluate")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--num_const", type=int, default=100, help="Number of constituents")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size")
    parser.add_argument("--bkg", type=str, default='qcd', choices=['qcd', 'top'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_features = 3
    num_bins = (41, 41, 41)

    print(f'Loading model from {args.model_dir+args.bkg}')
    model = load_model(args.model_dir+args.bkg, 'last')
    model.eval()
    model.to(device)

    scores = np.empty((0,))
    labels = np.empty((0,))
    nparts = np.empty((0,))
    for c in ['top', 'qcd']:
        print(c)
        data_path = os.path.join(args.data_path, f'test_{c}.h5')
        df = pd.read_hdf(data_path, 'discretized')
        x, padding_mask, bins = preprocess_dataframe(df,
            num_features=num_features,
            num_bins=num_bins,
            num_const=args.num_const,
            num_events=200000,
            to_tensor=True,
            )

        test_dataset = TensorDataset(x, padding_mask, bins)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            )

        for x, padding_mask, true_bin in tqdm(test_loader,
                total=len(test_loader),
                desc=f'Evaluating {data_path}'
                ):
            nparts = np.append(nparts, padding_mask.sum(-1))
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            true_bin = true_bin.to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logits = model(x, padding_mask)
                    perplexity = model.probability(logits,
                        padding_mask,
                        true_bin,
                        perplexity=True,
                        logarithmic=False
                        )
            scores = np.append(scores,
                perplexity.cpu().detach().numpy(),
                axis=0,
                )
            labels = np.append(labels,
                int(c!=args.bkg) * np.ones(len(x)),
                axis=0,
                )
        print(labels[-1])


    print(labels.shape)
    print(scores.shape)

    plt.hist(scores[labels==0],
        histtype='step',
        bins=50,
        density=True,
        label='Background',
        )
    plt.hist(scores[labels==1],
        histtype='step',
        bins=50,
        density=True,
        label='Signal',
        )
    plt.legend()
    plt.savefig('tmp1.png')
    plt.close('all')

    plt.hist(nparts[labels==0],
        histtype='step',
        bins=50,
        density=True,
        label='Background',
        )
    plt.hist(nparts[labels==1],
        histtype='step',
        bins=50,
        density=True,
        label='Signal',
        )
    plt.legend()
    plt.savefig('tmp2.png')

    np.savez(os.path.join(args.model_dir+args.bkg, 'predictions_perp.npz'),
        labels=labels,
        scores=scores,
        nparts=nparts,
        )
    auc = roc_auc_score(y_true=labels, y_score=scores)
    print(auc)
