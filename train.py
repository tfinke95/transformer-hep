import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from model import JetTransformer
from preprocess import preprocess_dataframe

from argparse import ArgumentParser
from tqdm import tqdm
import os
import pandas as pd

torch.multiprocessing.set_sharing_strategy('file_system')


def get_linear_scheduler():
    training_steps = args.num_epochs * len(train_loader)
    lr_fn = lambda step: 1.0 - ((1.0 - args.lr_decay) * (step / training_steps))
    scheduler = LambdaLR(opt, lr_lambda=lr_fn)
    return scheduler


def save_opt_states():
    torch.save(
        {
            'opt_state_dict': opt.state_dict(),
            'sched_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        },
        os.path.join(args.model_dir, 'opt_state_dict.pt')
    )


def load_opt_states():
    state_dicts = torch.load(os.path.join(args.model_dir, 'opt_state_dict.pt'))
    opt.load_state_dict(state_dicts['opt_state_dict'])
    scheduler.load_state_dict(state_dicts['sched_state_dict'])
    scaler.load_state_dict(state_dicts['scaler_state_dict'])
    return opt, scheduler, scaler


def save_model(name):
    torch.save(model, os.path.join(args.model_dir, f'model_{name}.pt'))


def load_model(name):
    model = torch.load(os.path.join(args.model_dir, f'model_{name}.pt'))
    return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='models/test', help="Model directory")
    parser.add_argument("--data_path", type=str, default='Datasets/train_top.h5', help="Path to training data file")

    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--logging_steps", type=int, default=10, help="Training steps between logging")
    parser.add_argument("--checkpoint_steps", type=int, default=5000, help="Training steps between saving checkpoints")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")

    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.01, help="learning rate decay (linear)")
    parser.add_argument("--weight_decay", type=float, default=0.00001, help="weight decay")

    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dim of the model")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--output", type=str, default='linear', choices=['linear', 'embprod'], help="Output function")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_features = 3
    num_bins = (41, 41, 41)

    # load and preprocess data
    df = pd.read_hdf(args.data_path, 'discretized')
    x, padding_mask, bins = preprocess_dataframe(df, num_features=num_features, num_bins=num_bins, to_tensor=True)

    train_dataset = TensorDataset(x, padding_mask, bins)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    # construct model
    model = JetTransformer(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_features=num_features,
        num_bins=num_bins,
        dropout=args.dropout,
        output=args.output
    )
    model.to(device)

    # construct optimizer and auto-caster
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_scheduler()
    scaler = torch.cuda.amp.GradScaler()

    logger = SummaryWriter(args.model_dir)
    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        loss_list = []
        perplexity_list = []

        for x, padding_mask, true_bin in tqdm(train_loader, total=len(train_loader), desc=f'Training Epoch {epoch + 1}'):
            opt.zero_grad()
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            true_bin = true_bin.to(device)

            with torch.cuda.amp.autocast():
                logits = model(x, padding_mask)
                loss = model.loss(logits, true_bin)
                with torch.no_grad():
                    perplexity = model.probability(logits, padding_mask, true_bin, perplexity=True, logarithmic=False)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)

            scaler.step(opt)
            scaler.update()
            scheduler.step()

            loss_list.append(loss.cpu().detach().numpy())
            perplexity_list.append(perplexity.mean().cpu().detach().numpy())

            if (global_step + 1) % args.logging_steps == 0:
                logger.add_scalar('Train/Loss', np.mean(loss_list), global_step)
                logger.add_scalar('Train/Perplexity', np.mean(perplexity_list), global_step)
                loss_list = []

            #if (global_step + 1) % args.checkpoint_steps == 0:
            #    save_model(f'checkpoint_{global_step}')

            global_step += 1

        save_model('last')
        #save_opt_states(model.model_dir)
