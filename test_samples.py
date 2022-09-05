# %%
from model import CNNclass
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from preprocess import imagePreprocessing

# %%
kind = 'qcd'
data1 = np.load(f'sampled_100k_{kind}.npy')
df = pd.read_hdf(f'/home/thorben/Data/jet_datasets/top_benchmark/v0/test_{kind}_30_bins.h5', 'discretized', stop=100000)
data2 = df.to_numpy(dtype=np.int64)[:, :60]
data2 = data2.reshape(data2.shape[0], -1, 3)

data = np.append(data1, data2, axis=0)
labels = np.append(np.zeros(len(data1)), np.ones(len(data2)))

# %%
jets = imagePreprocessing(data.astype(float))
images = np.zeros((len(jets), 30, 30))
for i in range(len(jets)):
    images[i], _, _ = np.histogram2d(jets[i, :, 1], jets[i, :, 2],
                                    bins=(np.arange(-15.5, 15.5, 1),
                                    np.arange(-15.5, 15.5, 1)),
                                    weights=jets[i, :, 0],)
images = images[:,np.newaxis, ...]
images.shape

# %%

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classi = CNNclass()
idcs = np.random.permutation(len(images))
images = images[idcs]
labels = labels[idcs]

train_images = torch.tensor(images[:int(0.9*len(images))], dtype=torch.float32)
train_labels = torch.tensor(labels[:int(0.9*len(images))], dtype=torch.float32)

val_images = torch.tensor(images[int(0.9*len(images)):], dtype=torch.float32)
val_labels = torch.tensor(labels[int(0.9*len(images)):], dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(train_images, train_labels[..., np.newaxis])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(val_images, val_labels[..., np.newaxis])
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256,)

# %%
classi.to(device)
opt = torch.optim.Adam(classi.parameters(), lr=0.001,)

losses = []
val_losses = []
for epoch in tqdm(range(2)):
    classi.train()
    for x, y in dataloader:
        opt.zero_grad()

        x = x.to(device)
        y = y.to(device)
                
        pred = classi(x)
        loss = classi.loss(pred, y)
        losses.append(loss.cpu().detach().numpy())

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
        val_losses.append(np.mean(tmp))
        print(val_losses[-1])


# %%
plt.plot(losses)

labels = []
preds = []
for x, y in val_dataloader:
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        pred = classi(x)
        preds.append(pred.cpu().detach().numpy())
        labels.append(y.cpu().detach().numpy())

preds = np.concatenate(preds, axis=0)
labels = np.concatenate(labels, axis=0)
preds.shape, labels.shape, len(val_dataloader)
# %%
fpr, tpr, _ = roc_curve(y_true=labels, y_score=preds)
auc = roc_auc_score(y_true=labels, y_score=preds)

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(tpr, 1. / fpr, label=f'AUC: {auc:.3f}')
ax.plot(np.linspace(0, 1, 1000), 1. / np.linspace(0, 1, 1000), color='grey')
ax.set_yscale('log')
ax.grid(which='both')
ax.set_ylim(0.9, 1e3)
ax.legend()

plt.show()
# %%
