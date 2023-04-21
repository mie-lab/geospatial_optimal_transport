import os
import matplotlib.pyplot as plt


def plot_error_evolvement(error_evolvement, out_path=None):
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    ax.plot(
        error_evolvement[:, 0],
        error_evolvement[:, 1],
        color="green",
        label="MAE",
    )
    ax.set_ylabel("MAE", color="green")
    ax2 = ax.twinx()
    ax2.plot(
        error_evolvement[:, 0],
        error_evolvement[:, 1] / error_evolvement[:, 0],
        color="blue",
        label="normed",
    )
    ax2.set_ylabel("Normalized MAE (per station)", color="blue")
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
