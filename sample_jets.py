import torch
import numpy as np
from tqdm import tqdm
import time, os
from argparse import ArgumentParser

torch.multiprocessing.set_sharing_strategy("file_system")


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, default="models/test")
parser.add_argument("--model_name", type=str, default="model_last.pt")
parser.add_argument("--savetag", type=str, default="test")
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--batchsize", type=int, default=100)
parser.add_argument("--num_const", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

set_seeds(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "Not running on GPU"

n_batches = args.num_samples // args.batchsize
rest = args.num_samples % args.batchsize

# Load model for sampling
model = torch.load(os.path.join(args.model_dir, args.model_name))
model.classifier = False
model.to(device)
model.eval()

jets = []
bins = []
start = time.time()
for i in tqdm(range(n_batches), total=n_batches, desc="Sampling batch"):
    _jets, _bins = model.sample(
        starts=torch.zeros((args.batchsize, 3), device=device),
        device=device,
        len_seq=args.num_const + 1,
    )
    jets.append(_jets.cpu().numpy())
    bins.append(_bins.cpu().numpy())

if rest != 0:
    _jets, _bins = model.sample(
        starts=torch.zeros((rest, 3), device=device),
        device=device,
        len_seq=51,
    )
    jets.append(_jets.cpu().numpy())
    bins.append(_bins.cpu().numpy())


jets = np.concatenate(jets, 0)
bins = np.concatenate(bins, 0)
print(jets.dtype)

np.savez(
    os.path.join(args.model_dir, f"samples_{args.savetag}"),
    jets=jets[:, 1:],
    bins=bins[:, 1:],
)
print(int(time.time() - start))
