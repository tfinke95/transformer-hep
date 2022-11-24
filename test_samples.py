# %%
from model import CNNclass
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from preprocess import imagePreprocessing

torch.manual_seed(0)
np.random.seed(0)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


# %%
kind = "qcd"
tag = "QCD_data-negSamples"
data1 = np.load(f"notebooks/samples_qcd_noTanh.npz")["samples"][:10000]
data2 = np.load(f"samples/samples_qcd.npz")["samples"][:10000]
# data2 = np.load(f"sampled_test_qcd.npy")[:10000]
# data2 = np.load(f"samples/sampled_top.npy")

# df = pd.read_hdf(f'/hpcwork/bn227573/top_benchmark/test_qcd_30_bins.h5', 'discretized', start=100000, stop=200000)
# data1 = df.to_numpy(dtype=np.int64)[:, :60]
# data1 = data1.reshape(data1.shape[0], -1, 3)

# df = pd.read_hdf(
#     f"/mnt/wsl/PHYSICALDRIVE1p1/thorben/Data/jet_datasets/top_benchmark/v0/test_qcd_30_bins.h5",
#     "discretized",
#     stop=10000,
# )
# data1 = df.to_numpy(dtype=np.int64)[:, :60]
# data1 = data1.reshape(data1.shape[0], -1, 3)

data = np.append(data1, data2, axis=0)
labels = np.append(np.zeros(len(data1)), np.ones(len(data2)))

# %%
jets = imagePreprocessing(data.astype(float))
images = np.zeros((len(jets), 30, 30))
for i in range(len(jets)):
    images[i], _, _ = np.histogram2d(
        jets[i, :, 1],
        jets[i, :, 2],
        bins=(np.arange(-15.5, 15.5, 1), np.arange(-15.5, 15.5, 1)),
        weights=jets[i, :, 0],
    )
images = images[:, np.newaxis, ...]
images.shape

# %%

device = "cuda" if torch.cuda.is_available() else "cpu"

classi = CNNclass()
idcs = np.random.permutation(len(images))
images = images[idcs]
labels = labels[idcs]

train_images = torch.tensor(images[: int(0.9 * len(images))], dtype=torch.float32)
train_labels = torch.tensor(labels[: int(0.9 * len(images))], dtype=torch.float32)

val_images = torch.tensor(images[int(0.9 * len(images)) :], dtype=torch.float32)
val_labels = torch.tensor(labels[int(0.9 * len(images)) :], dtype=torch.float32)

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
for epoch in tqdm(range(20)):
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
            torch.save(classi, f"classifier_samples_{kind}_{tag}_best.pt")

losses = np.array(losses)
val_losses = np.array(val_losses)
np.savez(f"training_{kind}_{tag}.npz", losses=losses, val_losses=val_losses)

# %%
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(losses[:, 0], losses[:, 1], alpha=0.2, c="b")
ax.plot(
    losses[99:, 0], moving_average(losses[:, 1], 100), alpha=1.0, label="Train", c="b"
)
ax.plot(val_losses[:, 0], val_losses[:, 1], label="Val")
ax.legend()
fig.savefig(f"loss_{kind}_{tag}.png")

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

torch.save(classi, f"classifier_samples_{kind}_{tag}.pt")
preds = np.concatenate(preds, axis=0)
labels = np.concatenate(labels, axis=0)
np.savez(f"preds_{kind}_{tag}_last.npz", labels=labels, preds=preds)

# %%
fpr, tpr, _ = roc_curve(y_true=labels, y_score=preds)
auc = roc_auc_score(y_true=labels, y_score=preds)

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(tpr, 1.0 / fpr, label=f"AUC: {auc:.3f}")
ax.plot(np.linspace(0, 1, 1000), 1.0 / np.linspace(0, 1, 1000), color="grey")

classi = torch.load(f"classifier_samples_{kind}_{tag}_best.pt")
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
np.savez(f"preds_{kind}_{tag}_best.npz", labels=labels, preds=preds)
preds.shape, labels.shape, len(val_dataloader)

fpr, tpr, _ = roc_curve(y_true=labels, y_score=preds)
auc = roc_auc_score(y_true=labels, y_score=preds)

ax.plot(tpr, 1.0 / fpr, label=f"AUC: {auc:.3f}")
ax.plot(np.linspace(0, 1, 1000), 1.0 / np.linspace(0, 1, 1000), color="grey")

ax.set_yscale("log")
ax.grid(which="both")
ax.set_ylim(0.9, 1e3)
ax.legend()
fig.savefig(f"roc_samples_{kind}_{tag}.png")
plt.show()

# %%
fig, ax = plt.subplots()
ax.hist(
    [np.tanh(preds[labels == 0]) / 2 + 0.5, np.tanh(preds[labels == 1]) / 2 + 0.5],
    bins=30,
    histtype="step",
)

# %%
fig, ax = plt.subplots()
ax.hist([preds[labels == 0], preds[labels == 1]], bins=100, histtype="step")
# %%
