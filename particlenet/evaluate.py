import matplotlib.pyplot as plt
import os, time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from data_loader import get_config, load_data
from graphnet import GraphNet
from sklearn.metrics import roc_auc_score, roc_curve

NETWOKPARAMS = {
    "activation": tf.keras.layers.LeakyReLU(0.1),
    "k": 16,
    "channels": [[64, 64, 64], [128, 128, 128], [256, 256, 256]],
    "classifier": [256, 128, 2],
    "dropout": 0.1,
    "static": False,
}
TEST = True


def plot_trainHistory(folder):
    fig, ax = plt.subplots(constrained_layout=True)
    max_ep = 0
    i = 0
    colors = plt.cm.jet(np.linspace(0, 1, 10))
    for fold, _, files in os.walk(folder):
        if "f2_0.5" in fold and (("250000" in fold) or ("125000" in fold)):
            if "training.npz" in files:
                fit = np.load(os.path.join(fold, "training.npz"))
                epochs = len(fit["loss"])
                if epochs > max_ep:
                    max_ep = epochs
                ax.plot(
                    np.arange(1, epochs + 1),
                    fit["loss"],
                    c=colors[i],
                    label=f"{fold.split('/')[-3].split('_')[-1]}",
                )
                ax.plot(
                    np.arange(1, epochs + 1),
                    fit["val_loss"],
                    c=colors[i],
                    linestyle="--",
                )
                i += 1
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(np.arange(0, max_ep + 1, 5))
    # fig.savefig(os.path.join(folder, 'training.pdf'))
    plt.show()


def plot_roc(predictions, labels, folder, plot=False):
    fpr, tpr, _ = roc_curve(labels, predictions[:, 1])
    auc = roc_auc_score(labels, predictions[:, 1])
    print(auc)

    np.savez(
        os.path.join(folder, "predictions.npz"), predictions=predictions, labels=labels
    )
    np.savez(
        os.path.join(folder, "roc.npz"),
        fpr=fpr,
        tpr=tpr,
        auc=auc,
    )

    if plot:
        fig, ax = plt.subplots()
        ax.plot(tpr, 1.0 / fpr, label=f"{auc:.3f}")
        ax.plot(
            np.linspace(1e-5, 1, 100),
            1.0 / np.linspace(1e-5, 1, 100),
            linestyle="--",
            color="grey",
        )
        ax.legend(title="AUC")
        ax.set_yscale("log")
        ax.set_ylim(1, 6e3)
        ax.grid(which="both")
        fig.savefig(os.path.join(folder, f"roc_{'test' if TEST else 'train'}.pdf"))
        plt.close(fig)


def check_weights(model, folder, data, load=True):
    if load:
        model.load_weights(os.path.join(folder, "model_weights.h5"))
    check = np.load(os.path.join(folder, "check.npy"))
    pred = model.predict([np.ones_like(data[:10]), np.ones_like(data[:10])])
    diff = pred - check
    assert (
        np.sum(np.abs(diff) > 5e-7) == 0
    ), f"Differences in check predictions\n{diff[0]}\n{check[0]}\n{pred[0]}"
    return model


def main():
    model = GraphNet(**NETWOKPARAMS)
    config = get_config(test=True)
    data, labels = load_data(
        config["data"],
        test=TEST,
        plot_dists=os.path.join(config["logging"]["logfolder"], "dists.png"),
    )
    print(f"{data.shape=} {labels.shape=}")
    model([data[:2, :, :2], data[:2]])
    model = check_weights(model, config["logging"]["logfolder"], data=data)
    if config["mask"]:
        data = [data[:, :, :2], data, data[:, :, 2] != 0]
    else:
        data = [data[:, :, :2], data]
    preds = model.predict(data, batch_size=1024, verbose=1)
    plot_roc(
        predictions=preds,
        labels=labels,
        folder=config["logging"]["logfolder"],
        plot=True,
    )
    np.savez(
        os.path.join(
            config["logging"]["logfolder"], f"predictions_{'test' if TEST else 'train'}"
        ),
        predictions=preds,
        labels=labels,
    )

    pass


if __name__ == "__main__":
    main()
