import os
import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns

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


def plot_emd(path, out_path="figures"):
    subset = [f for f in os.listdir(path) if "150" in f and "csv" in f]
    subset_name_mapping = {
        "1_24_1_nhits_multi_70_3_3_1_basic_1_150_kmeans": "Kmeans hierarchical\n reconciled",
        "0_24_1_nhits_multi_70_3_3_0_basic_1_150_agg": "agglomerative",
        "0_24_1_nhits_multi_70_3_3_0_combined_sinkhorn_1_150_kmeans": "sinkhorn\n (Kmeans)",
        "1_24_1_nhits_multi_70_3_3_0_basic_1_150_kmeans": "Kmeans \n hierarchical",
        "0_24_1_nhits_multi_70_3_3_0_crossentropy_1_150_agg": "cross-entropy\n (agglomerative)",
        "0_24_1_nhits_multi_70_3_3_0_distribution_1_150_agg": "distribution \n(agglomerative)",
        "0_24_1_nhits_multi_70_3_3_0_basic_1_150_kmeans": "Kmeans",
    }

    gt_file = [f for f in os.listdir(path) if "None" in f][0]
    # load gt as reference (per station gt needed for evaluation)
    gt_reference = pd.read_csv(os.path.join(path, gt_file)).drop("pred", axis=1)
    # load stations
    stations = (
        pd.read_csv("data/bikes_montreal/test_stations.csv")
        .sort_values("station_id")
        .set_index("station_id")
    )

    emd_res_dict = {}
    for f in subset:
        print(f)
        res = pd.read_csv(os.path.join(path, f))
        with open(
            os.path.join(path, f[:-4] + "_hierarchy.json"), "r"
        ) as infile:
            res_hierarchy = json.load(infile)

        if f.startswith("1"):
            # reduce to groups
            res = res[res["group"].str.contains("Group")]

        emd_compute = EMDWrapper(stations, gt_reference)
        emd_vals = emd_compute(res, res_hierarchy, mode="group_to_group")
        emd_res_dict[subset_name_mapping[f[:-4]]] = emd_vals

    # simplify the data (just plot the average)
    emd_plot_df = pd.DataFrame(
        pd.DataFrame(emd_res_dict).mean(), columns=["EMD"]
    ).sort_values("EMD")
    emd_plot_df.index.name = "Method"
    emd_plot_df.reset_index(inplace=True)

    # plot
    plt.figure(figsize=(20, 5))
    sns.barplot(data=emd_plot_df, x="Method", y="EMD")
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "emd_comparison.pdf"))
