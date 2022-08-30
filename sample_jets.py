import torch
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
from argparse import ArgumentParser

torch.multiprocessing.set_sharing_strategy('file_system')

def idx_to_bins(x):
    pT = x % 41
    eta = (x - pT) // 41 % (1271 // 41)
    phi = (x - pT - 41 * eta) // 1271
    return pT, eta, phi


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, default='models/test')
parser.add_argument("--data_path", type=str, default='/home/thorben/Data/jet_datasets/top_benchmark/v0/test_top_30_bins.h5')
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)


args = parser.parse_args()

set_seeds(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model for sampling
model = torch.load(args.model_dir)
model.to(device)

# Load initial particles for starting point
df = pd.read_hdf(args.data_path, 'discretized', stop=100000)
x = df.to_numpy(dtype=np.int64)[:, :3]
x = x.reshape(x.shape[0], -1, 3)
x = torch.tensor(x, dtype=torch.long,)


start = time.time()
njets = args.num_samples
jets = -torch.empty((njets, 20, 3), dtype=torch.long)
softmax = torch.nn.Softmax(dim=-1)

# Sample jets jet by jet
with torch.no_grad():
    for jet_idx in tqdm(range(njets)):
        # Set first particle to one of the true jets
        current_jet = - torch.ones((1, 20, 3), dtype=torch.long, device='cuda')
        current_jet[:, 0] = x[torch.randint(len(x), (1,))]

        # Set padding to ignore all particles not generated yet
        padding_mask = current_jet[:, :, 0] != -1
        padding_mask.to('cuda')
        
        for particle in range(19):
            # Get probability predictions
            preds = model(current_jet, padding_mask)
            preds = softmax(preds[:, :-1])
            rand = torch.rand((1,), device='cuda')

            # Sample the bin by checking the cumsum to be larger than random value
            preds_cum = torch.cumsum(preds[0, particle], dim=-1)
            idx = torch.searchsorted(preds_cum, rand,)
            bins = idx_to_bins(idx)

            for ind, tmp_bin in enumerate(bins):
                current_jet[0, particle+1, ind] = tmp_bin

            # Update padding
            padding_mask = current_jet[:, :, 0] != -1
            

        jets[jet_idx] = current_jet[0]

np.save('sampled_qcd.npy', jets)
print(int(time.time() - start))
