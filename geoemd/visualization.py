import os
import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns
from sklearn.cluster import KMeans

from geoemd.emd_eval import EMDWrapper


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


def plot_stations(
    out_path, station_path="data/bikes_montreal/test_stations.csv"
):
    stations = pd.read_csv(station_path).set_index("station_id")

    plt.scatter(stations["x"], stations["y"])
    plt.xlabel("Projected x coordinate")
    plt.ylabel("Projected y coordinate")
    plt.savefig(os.path.join(out_path, "all_stations.csv"))

    for k in [5, 10, 20, 40]:
        clustering = KMeans(k)
        clustering.fit(stations)
        plt.scatter(stations["x"], stations["y"], c=clustering.labels_)
        plt.xlabel("Projected x coordinate")
        plt.ylabel("Projected y coordinate")
        plt.savefig(os.path.join(out_path, f"stations_clustered_{k}.csv"))


def plot_single_group_ordered(path_ordered_out_file, out_path, horizon=2):
    out = pd.read_csv(path_ordered_out_file)
    # version 1: using always the three predictions, then redo prediction
    # out["timeslot"] = out["val_sample_ind"] + out["steps_ahead"]
    # version 2: using always the three steps ahead prediction
    out = out[out["steps_ahead"] == horizon]
    out["timeslot"] = out["val_sample_ind"]

    groups = out["group"].unique()
    print("Plotting over time for group", groups[0])
    plot_out = out[out["group"] == groups[0]]

    plt.figure(figsize=(8, 5))
    plt.plot(plot_out["timeslot"], plot_out["gt"], label="gt")
    plt.plot(plot_out["timeslot"], plot_out["pred"], label="pred")
    plt.legend()
    plt.savefig(out_path)


if __name__ == "__main__":
    path = "outputs/ordered_04_08"
    for f in os.listdir(path):
        if f[-3:] != "csv":
            continue
        plot_single_group_ordered(
            os.path.join(path, f), os.path.join(path, f[:-4] + "_plot.png")
        )
