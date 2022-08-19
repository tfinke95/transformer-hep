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


def get_lin_scheduler():
    training_steps = args.num_epochs * len(train_loader)
    lr_fn = lambda step: 1.0 - ((1.0 - args.lr_decay) * (step / training_steps))
    scheduler = LambdaLR(opt, lr_lambda=lr_fn)
    return scheduler


def get_exp_scheduler(final_reduction=1e-2):
    training_steps = args.num_epochs * len(train_loader)
    lr_fn = lambda step: final_reduction ** (step / training_steps)
    scheduler = LambdaLR(opt, lr_lambda=lr_fn)
    return scheduler


def get_cos_scheduler():
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt,
        # T_0=len(train_loader)//5,
        T_0=len(train_loader) * args.num_epochs + 1,
        eta_min=1e-6
    )
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


def save_model(name):
    torch.save(model, os.path.join(args.model_dir, f'model_{name}.pt'))


def load_model(name):
    model = torch.load(os.path.join(args.model_dir, f'model_{name}.pt'))
    return model


def parse_input():
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='models/test', help="Model directory")
    parser.add_argument("--data_path", type=str, default='/hpcwork/bn227573/top_benchmark/train_qcd_30_bins.h5', help="Path to training data file")

    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--logging_steps", type=int, default=10, help="Training steps between logging")
    parser.add_argument("--checkpoint_steps", type=int, default=5000, help="Training steps between saving checkpoints")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")

    parser.add_argument("--num_const", type=int, default=100, help="Number of constituents")
    parser.add_argument("--limit_const", action="store_true", help="Only use jets with at least num_const constituents")
    parser.add_argument("--num_events", type=int, default=10000, help="Number of events for training")
    parser.add_argument("--num_bins", type=int, nargs=3, default=[41, 31, 31], help="Number of bins per feature")
    parser.add_argument("--contin", action='store_true', help="Whether to continue training")
    parser.add_argument("--global_step", type=int, default=0, help="Starting point of step counter")
    parser.add_argument("--reverse", action='store_true', help="Whether to reverse pt order")

    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
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
    return args


def save_arguments(args):
    if not os.path.isdir(args.model_dir): os.makedirs(args.model_dir)
    with open(os.path.join(args.model_dir, 'arguments.txt'), 'w') as f:
        arg_dict= vars(args)
        for k,v in arg_dict.items():
            f.write(f'{k:20s} {v}\n')


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_data(path, n_events):
    df = pd.read_hdf(path, 'discretized', stop=n_events)
    x, padding_mask, bins = preprocess_dataframe(df, num_features=num_features,
                                num_bins=num_bins,
                                to_tensor=True,
                                num_const=args.num_const,
                                reverse=args.reverse,
                                limit_nconst=args.limit_const)

    train_dataset = TensorDataset(x, padding_mask, bins)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    print(x.shape)
    return train_loader


if __name__ == '__main__':
    args = parse_input()
    save_arguments(args)

    set_seeds(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    num_features = 3
    num_bins = tuple(args.num_bins)

    print(f"Using bins: {num_bins}")
    print(f"{'Not' if not args.reverse else ''} reversing pt order")

    # load and preprocess data
    print(f"Loading training set")
    train_loader = load_data(args.data_path, args.num_events)

    print("Loading validation set")
    val_loader = load_data(args.data_path.replace('train', 'test'),
                            10000)

    # construct model
    if args.contin:
        model = load_model('last')
        print("Loaded model")
    else:
        model = JetTransformer(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_features=num_features,
            num_bins=num_bins,
            dropout=args.dropout,
            output=args.output,
        )
    model.to(device)

    # construct optimizer and auto-caster
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cos_scheduler()
    scaler = torch.cuda.amp.GradScaler()

    if args.contin:
        load_opt_states()
        print("Loaded optimizer")

    logger = SummaryWriter(args.model_dir)
    global_step = args.global_step
    loss_list = []
    perplexity_list = []
    for epoch in range(args.num_epochs):
        model.train()

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
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            loss_list.append(loss.cpu().detach().numpy())
            perplexity_list.append(perplexity.mean().cpu().detach().numpy())

            if (global_step + 1) % args.logging_steps == 0:
                logger.add_scalar('Train/Loss', np.mean(loss_list), global_step)
                logger.add_scalar('Train/Perplexity', np.mean(perplexity_list), global_step)
                logger.add_scalar('Train/LR', scheduler.get_last_lr()[0], global_step)
                loss_list = []
                perplexity_list = []

            #if (global_step + 1) % args.checkpoint_steps == 0:
            #    save_model(f'checkpoint_{global_step}')

            global_step += 1

        model.eval()
        with torch.no_grad():
            val_loss = []
            val_perplexity = []
            for x, padding_mask, true_bin in tqdm(val_loader, total=len(val_loader), desc=f'Validation Epoch {epoch + 1}'):
                x = x.to(device)
                padding_mask = padding_mask.to(device)
                true_bin = true_bin.to(device)

                logits = model(x, padding_mask,)
                loss = model.loss(logits, true_bin)
                perplexity = model.probability(logits, padding_mask, true_bin, perplexity=True, logarithmic=False)
                val_loss.append(loss.cpu().detach().numpy())
                val_perplexity.append(perplexity.mean().cpu().detach().numpy())

            logger.add_scalar('Val/Loss', np.mean(val_loss), global_step)
            logger.add_scalar('Val/Perplexity', np.mean(val_perplexity), global_step)

        save_model('last')
        save_opt_states()
