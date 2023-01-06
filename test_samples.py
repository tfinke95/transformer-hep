# %%
from model import CNNclass
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from preprocess import imagePreprocessing
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--bg", "-b", type=str)
parser.add_argument("--sig", "-s", type=str)
parser.add_argument("--save_dir", type=str, default="samples_test/test")
parser.add_argument("--num_jets", "-N", type=int, default=100000)
parser.add_argument("--num_const", "-c", type=int, default=50)
parser.add_argument("--num_epochs", "-E", type=int, default=30)
args = parser.parse_args()

if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)

torch.manual_seed(0)
np.random.seed(0)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


# %%
def load_data(file):
    if file.endswith("npz"):
        dat = np.load(file)["jets"][: args.num_jets, : args.num_const]
    elif file.endswith("h5"):
        dat = pd.read_hdf(file, key="discretized", stop=args.num_jets)
        dat = dat.to_numpy(dtype=np.int64)[:, : args.num_const * 3]
        dat = dat.reshape(dat.shape[0], -1, 3)
    else:
        assert False, "Filetype for bg not supported"
    dat = np.delete(dat, np.where(dat[:, 0, 0] == 0)[0], axis=0)
    return dat


bg = load_data(args.bg)
print(f"Got bg from {args.bg}")
sig = load_data(args.sig)
print(f"Got sig from {args.sig}")

print(bg.shape, sig.shape)
# %%

data = np.append(bg, sig, axis=0)
data[data == -1] = 0
labels = np.append(np.zeros(len(bg)), np.ones(len(sig)))

# %%
jets = imagePreprocessing(data.astype(float))
images = np.zeros((len(jets), 30, 30))
for i in tqdm(range(len(jets))):
    images[i], _, _ = np.histogram2d(
        jets[i, :, 1],
        jets[i, :, 2],
        bins=(np.arange(-15.5, 15.5, 1), np.arange(-15.5, 15.5, 1)),
        weights=jets[i, :, 0],
    )
images = images[:, np.newaxis, ...]

fig, ax = plt.subplots(1, 2, constrained_layout=True)
for i in range(2):
    ax[i].imshow(images[labels == i].mean(0).squeeze(), cmap="Blues")
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_title(["Bg", "Sig"][i])
fig.savefig(os.path.join(args.save_dir, "mean_images.png"))


# %%

device = "cuda" if torch.cuda.is_available() else "cpu"

classi = CNNclass()
idcs = np.random.permutation(len(images))
images = images[idcs]
labels = labels[idcs]

train_images = torch.tensor(images[: int(0.75 * len(images))], dtype=torch.float32)
train_labels = torch.tensor(labels[: int(0.75 * len(images))], dtype=torch.float32)

val_images = torch.tensor(images[int(0.75 * len(images)) :], dtype=torch.float32)
val_labels = torch.tensor(labels[int(0.75 * len(images)) :], dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(train_images, train_labels[..., np.newaxis])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(val_images, val_labels[..., np.newaxis])
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=256,
)

# %%
classi.to(device)
opt = torch.optim.Adam(
    classi.parameters(),
    lr=0.001,
)

losses = []
val_losses = []
min_loss = np.inf
global_step = 0
for epoch in tqdm(range(args.num_epochs)):
    classi.train()
    for x, y in dataloader:
        global_step += 1
        opt.zero_grad()

        x = x.to(device)
        y = y.to(device)

        pred = classi(x)
        loss = classi.loss(pred, y)
        losses.append([global_step, loss.cpu().detach().numpy()])

        loss.backward()
        opt.step()

    with torch.no_grad():
        classi.eval()
        tmp = []
        for x, y in val_dataloader:
            x = x.to(device)
            y = y.to(device)
            loss = classi.loss(classi(x), y)
            tmp.append(loss.cpu().detach().numpy())
        val_losses.append([global_step, np.mean(tmp)])
        if np.mean(tmp) < min_loss:
            min_loss = np.mean(tmp)
            torch.save(classi, os.path.join(args.save_dir, "classifier_best.pt"))

losses = np.array(losses)
val_losses = np.array(val_losses)
np.savez(os.path.join(args.save_dir, "training"), losses=losses, val_losses=val_losses)

# %%
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(losses[:, 0], losses[:, 1], alpha=0.2, c="b")
ax.plot(
    losses[99:, 0], moving_average(losses[:, 1], 100), alpha=1.0, label="Train", c="b"
)
ax.plot(val_losses[:, 0], val_losses[:, 1], label="Val")
ax.legend()
fig.savefig(os.path.join(args.save_dir, "training.png"))

# %%
labels = []
preds = []
classi.eval()
for x, y in val_dataloader:
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        pred = classi(x)
        preds.append(pred.cpu().detach().numpy())
        labels.append(y.cpu().detach().numpy())

torch.save(classi, os.path.join(args.save_dir, "classifier_last.pt"))
preds = np.concatenate(preds, axis=0)
labels = np.concatenate(labels, axis=0)
np.savez(os.path.join(args.save_dir, "predictions_last"), labels=labels, preds=preds)

# %%
fpr, tpr, _ = roc_curve(y_true=labels, y_score=preds)
auc = roc_auc_score(y_true=labels, y_score=preds)

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(tpr, 1.0 / fpr, label=f"Last AUC: {auc:.3f}")
ax.plot(np.linspace(0, 1, 1000), 1.0 / np.linspace(0, 1, 1000), color="grey")

classi = torch.load(os.path.join(args.save_dir, "classifier_best.pt"))
labels = []
preds = []
classi.eval()
for x, y in val_dataloader:
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        pred = classi(x)
        preds.append(pred.cpu().detach().numpy())
        labels.append(y.cpu().detach().numpy())

preds = np.concatenate(preds, axis=0)
labels = np.concatenate(labels, axis=0)
np.savez(os.path.join(args.save_dir, "predictions_best"), labels=labels, preds=preds)

fpr, tpr, _ = roc_curve(y_true=labels, y_score=preds)
auc = roc_auc_score(y_true=labels, y_score=preds)

ax.plot(tpr, 1.0 / fpr, label=f"Best AUC: {auc:.3f}")
ax.plot(np.linspace(0, 1, 1000), 1.0 / np.linspace(0, 1, 1000), color="grey")

ax.set_yscale("log")
ax.grid(which="both")
ax.set_ylim(0.9, 1e5)
ax.legend()
fig.savefig(os.path.join(args.save_dir, "roc.png"))

# %%

fig, ax = plt.subplots(constrained_layout=True)
ax.hist([preds[labels == 0], preds[labels == 1]], bins=100, histtype="step")
ax.set_yscale("log")
fig.savefig(os.path.join(args.save_dir, "predictions.png"))
