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
    parser.add_argument("--data_path", type=str, default='/hpcwork/bn227573/top_benchmark/', help="Path to training data file")
    parser.add_argument("--data_split", type=str, default='test', help="Split to evaluate")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--num_const", type=int, default=100, help="Number of constituents")
    parser.add_argument("--num_events", type=int, default=200000, help="Number of events")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size")
    parser.add_argument("--bkg", type=str, default='qcd', choices=['qcd', 'top'])
    parser.add_argument("--reverse", action="store_true", help="Reverse the pt ordering")
    parser.add_argument("--kind", type=str, default='perp', choices=['perp', 'log'], help='Kind of anomaly score')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on device {device}')

    if args.kind == 'perp':
        perp, log = True, False
    elif args.kind == 'log':
        perp, log = False, True

    num_features = 3
    num_bins = (41, 31, 31)

    print(f'Loading model from {args.model_dir+args.bkg}')
    model = load_model(args.model_dir+args.bkg, 'last')
    model.eval()
    model.to(device)

    scores = np.empty((0,))
    labels = np.empty((0,))
    nparts = np.empty((0,), dtype=int)
    losses = np.empty((0, 99))
    stats = np.empty((0, 100, 4))
    for c in ['qcd', 'top',]:
        print(c)
        tmp_losses = np.empty((0, 99))
        data_path = os.path.join(args.data_path, f'test_{c}_30_bins.h5')
        df = pd.read_hdf(data_path, 'discretized')
        x, padding_mask, bins = preprocess_dataframe(df,
            num_features=num_features,
            num_bins=num_bins,
            num_const=args.num_const,
            num_events=args.num_events,
            to_tensor=True,
            reverse=args.reverse
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
                    probs = torch.nn.Softmax(dim=-1)(logits)
                    prob_min = probs.min(-1).values.cpu().detach().numpy()
                    prob_max = probs.max(-1).values.cpu().detach().numpy()
                    prob_mean = probs.mean(-1).cpu().detach().numpy()
                    prob_median = torch.median(probs, -1).values.cpu().detach().numpy()
                    stat = np.stack((prob_min, prob_max, prob_mean, prob_median), axis=-1)
                    stats = np.append(stats, stat, axis=0)

                    loss = model.loss_pC(logits, true_bin)
                    tmp_losses = np.append(tmp_losses,
                        loss.reshape(-1, 99).cpu().detach().numpy(),
                        axis=0)
                    perplexity = model.probability(logits,
                        padding_mask,
                        true_bin,
                        perplexity=perp,
                        logarithmic=log
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
        tmp_losses[bins[:, 1:] == -100] = np.nan
        losses = np.append(losses, tmp_losses, axis=0)
        
    print(losses.shape)
    print(labels.shape)
    print(scores.shape)
    print(stats.shape)

    np.save('tmp', stats)
    np.savez(os.path.join(args.model_dir+args.bkg, f'predictions_{args.kind}.npz'),
        labels=labels,
        scores=scores,
        nparts=nparts,
        losses=losses,
        probs=stats,
        )
    auc = roc_auc_score(y_true=labels, y_score=scores)
    print(auc)
