import torch
from argparse import ArgumentParser
from helpers_train import set_seeds, load_data
import numpy as np
from tqdm import tqdm


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model")
    parser.add_argument("--data", type=str, help="Path to data")
    parser.add_argument(
        "--num_const",
        type=int,
        default=50,
        help="Number of constituents per jet. Default 50",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for torch and numpy. Default 0",
    )
    parser.add_argument(
        "--num_events",
        type=int,
        default=None,
        help="Number of jets used. Defaults to all available in 'data'",
    )

    return parser.parse_args()


def get_probs(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)
    model.eval()
    probs = []
    for x, mask, bins in tqdm(loader, total=len(loader)):
        with torch.no_grad():
            x, mask, bins = x.to(device), mask.to(device), bins.to(device)
            logits = model.forward(x, mask)
            probability = model.probability(logits, mask, bins, logarithmic=True)
            probs.append(probability.cpu().numpy())

    return np.concatenate(probs, 0)


def main():
    args = get_args()
    set_seeds(args.seed)
    model = torch.load(args.model, map_location="cpu")
    loader = load_data(
        args.data,
        args.num_events,
        start_token=True,
        end_token=True,
        limit_const=False,
        num_const=args.num_const,
    )
    probs = get_probs(model=model, loader=loader)
    import matplotlib.pyplot as plt

    print(args.model.split("/")[:-1])
    plt.hist(probs, bins=30)
    plt.show()
    print(probs.shape)


if __name__ == "__main__":
    main()
