import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model import JetTransformer
from preprocess import preprocess_dataframe

from argparse import ArgumentParser
from tqdm import tqdm
import os
import pandas as pd
from glob import glob

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
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_features = 3
    num_bins = (41, 41, 41)
    classes = ['top', 'qcd']

    # load model
    models = {}
    for c in classes:
        # model_dir = os.path.join(args.model_dir, c)
        model_dir = args.model_dir + c
        print(model_dir)
        model = load_model(model_dir, args.model_name)
        model.eval()
        model.to(device)
        models[c] = model

    perplexity_list = []
    y_list = []
    for i, data_class in enumerate(classes):
        data_path = os.path.join(args.data_path, f'{args.data_split}_{data_class}.h5')
        df = pd.read_hdf(data_path, 'discretized')
        x, padding_mask, bins = preprocess_dataframe(df, num_features=num_features, num_bins=num_bins, num_const=40, num_events=1000, to_tensor=True)

        test_dataset = TensorDataset(x, padding_mask, bins)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        for x, padding_mask, true_bin in tqdm(test_loader, total=len(test_loader), desc=f'Evaluating {data_path}'):
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            true_bin = true_bin.to(device)

            batch_perplexity = []
            for model_class in classes:
                model = models[model_class]
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        logits = model(x, padding_mask)
                        perplexity = model.probability(logits, padding_mask, true_bin, perplexity=False, logarithmic=True)
                batch_perplexity.append(perplexity.cpu().detach().numpy().reshape(-1, 1))

            batch_perplexity = np.hstack(batch_perplexity)
            batch_y = np.int64([i]*x.shape[0])
            perplexity_list.append(batch_perplexity)
            y_list.append(batch_y)

    perplexity = np.vstack(perplexity_list)
    score_direct = -perplexity[:,1]
    score_reverse = -perplexity[:,0]
    y = np.concatenate(y_list, axis=0)
    auc_direct = roc_auc_score(y_true=np.abs(y-1), y_score=score_direct)
    auc_reverse = roc_auc_score(y_true=y, y_score=score_reverse)
    print(auc_direct)
    print(auc_reverse)
    exit()




    score = perplexity[:, 1] / (2 * perplexity.sum(axis=1))
    acc = np.float32(perplexity.argmax(axis=1) == y).mean()
    auc = roc_auc_score(y_true=y, y_score=score)
    fpr, tpr, _ = roc_curve(y_true=y, y_score=score)
    # np.savez('tmp.npz', fpr=fpr, tpr=tpr)
    print(f'Accuracy : {acc}, AUC: {auc}')
