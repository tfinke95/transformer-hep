from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def plot_scores():
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig1, ax1 = plt.subplots(2, 1, sharex=True)
    kind = 'perp'
    print(kind)
    data = np.load(f'models/N600k_E50_nC100_qcd/predictions_{kind}.npz')
    scores = data['scores']
    labels = data['labels']
    upper = np.quantile(scores, 0.995)
    lower = np.quantile(scores, 0.005)
    idx = np.random.permutation(len(scores))
    scores = scores[idx]
    labels = labels[idx]

    ax[0].hist(scores[labels==0],
               density=True, bins=50, histtype='step', range=[lower, upper],
               label='QCD')
    ax[0].hist(scores[labels==1],
               density=True, bins=50, histtype='step', range=[lower, upper],
               label='Top')
    ax[0].legend()
    ax[0].set_title('Trained on qcd')

    ax1[0].scatter(scores, data['nparts'][idx],
                   c=labels, s=1, alpha=0.1,)
    ax1[0].set_xlim(lower, upper)
    ax1[0].set_title('Trained on qcd')

    data = np.load(f'models/N600k_E50_nC100_top/predictions_{kind}.npz')
    scores = data['scores']
    labels = data['labels']
    idx = np.random.permutation(len(scores))
    scores = scores[idx]
    labels = labels[idx]
    ax[1].hist(scores[labels==1],
               density=True, bins=50, histtype='step', range=[lower, upper],
               label='QCD')
    ax[1].hist(scores[labels==0],
               density=True, bins=50, histtype='step', range=[lower, upper],
               label='Top')
    ax[1].set_xlabel('score')
    ax[1].set_title('Trained on top')

    ax1[1].scatter(scores, data['nparts'][idx],
                   c=np.abs(labels-1), s=1, alpha=0.1,)
    ax1[1].set_title('Trained on top')
    ax1[1].set_xlim(lower, upper)

    fig.savefig(f'scores_{kind}.png')
    fig1.savefig(f'nParts_score_{kind}.png')


def plot_rocs():
    fig, ax = plt.subplots()
    score_kinds = ['perp']
    for score_kind in score_kinds:
        data = np.load(f'models/N600k_E50_nC100_qcd/predictions_{score_kind}.npz')
        scores = data['scores']
        labels = data['labels']
        fpr, tpr, _ = roc_curve(y_true=labels, y_score=-scores)

        ax.plot(tpr, 1. / fpr,
            label=f'nC 100 auc {roc_auc_score(labels, -scores):.3f} {score_kind}')

        data = np.load(f'models/N600k_E50_nC100_qcd/predictions_{score_kind}_40.npz')
        scores = data['scores']
        labels = data['labels']
        fpr, tpr, _ = roc_curve(y_true=labels, y_score=-scores)

        ax.plot(tpr, 1. / fpr,
            label=f'nC 40 auc {roc_auc_score(labels, -scores):.3f} {score_kind}')

    ax.plot(np.linspace(0, 1, 1000), 1. / np.linspace(0, 1, 1000),
        linestyle='--', color='grey')

    ax.grid(which='both')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim(0.9, 1e3)
    ax.set_title(f'Trained on QCD with 100 constituents')
    fig.savefig(f'roc_100_40.png')


if __name__ == '__main__':
    # plot_scores()
    plot_rocs()
