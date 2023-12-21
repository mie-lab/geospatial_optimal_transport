import os
import json
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 20})

from geoemd.emd_eval import EMDWrapper
from geoemd.utils import get_dataset_name
from geoemd.config import STATION_PATH, DATA_PATH, TRAINTEST_SPLIT
from geoemd.hierarchy.hierarchy_utils import add_fraction_to_hierarchy


def get_singlestations_file(path):
    gt_file = [f for f in os.listdir(path) if "None" in f][0]
    # load gt as reference (per station gt needed for evaluation)
    gt_reference = pd.read_csv(os.path.join(path, gt_file))
    return gt_reference


def load_stations(dataset):
    station_path = STATION_PATH[dataset]
    return (
        pd.read_csv(station_path)
        .sort_values("station_id")
        .set_index("station_id")
    )


def compare_emd(
    path,
    filter_step=-1,
    emd_mode="station_to_station",
    split_by_fraction=True,
    out_path=None,
):
    gt_reference = get_singlestations_file(path).drop("pred", axis=1)
    # load stations
    stations = load_stations(DATASET)
    if filter_step > 0:
        gt_reference = gt_reference[gt_reference["steps_ahead"] == filter_step]
    # if we want to add the fraction, we need to load the raw training data
    if split_by_fraction:
        data = pd.read_csv(DATA_PATH[DATASET])
        # pivot
        if "station_id" in data.columns:
            data = data.pivot(
                index="timeslot", columns="station_id", values="count"
            ).fillna(0)
        else:
            data = data.set_index("timeslot")
            data.columns = data.columns.astype(int)
        # reduce to training data
        data = data[: int(TRAINTEST_SPLIT * len(data))]

    emd_results = []
    for f in os.listdir(path):
        if f[-3:] != "csv" or f == "gt.csv" or f[0] == "1":
            continue
        loss_fn = f.split("_")[-4]
        # if loss_fn != "basic":
        #     continue
        print(f[:-4])
        # load file
        res = pd.read_csv(os.path.join(path, f))
        assert (
            res.iloc[0]["val_sample_ind"]
            == gt_reference.iloc[0]["val_sample_ind"]
        ), "GT samples do not correspond to pred samples"
        # filter for steps ahead if desired
        if filter_step > 0:
            res = res[res["steps_ahead"] == filter_step]

        # compute basic MAE errors
        res["error"] = (res["gt"] - res["pred"]).abs()

        # retrieve clustering method and groups from the file name
        clustering_method = [f[:-4].split("_")[-1]]
        nr_groups = int(f.split("_")[-2])
        if clustering_method[0] == "None":
            clustering_method = ["agg", "kmeans"]  # add both
            nr_groups = len(stations)
            res_hierarchy = None
            res["station_wise_error"] = res["error"]
        else:
            # load hierarchy
            with open(
                os.path.join(path, f[:-4] + "_hierarchy.json"), "r"
            ) as infile:
                res_hierarchy = json.load(infile)
            # split by fraction
            if split_by_fraction:
                res_hierarchy = add_fraction_to_hierarchy(res_hierarchy, data)
            # compute number of stations per group
            nr_stations_per_group = pd.Series(
                {k: len(v) for k, v in res_hierarchy.items()}
            )
            nr_stations_per_group.name = "nr_of_stations"
            # compute station-wise error
            res = res.merge(
                nr_stations_per_group,
                how="left",
                left_on="group",
                right_index=True,
            )
            res["station_wise_error"] = res["error"] / res["nr_of_stations"]

        # compute EMD
        emd_compute = EMDWrapper(
            stations, gt_reference, res_hierarchy, mode=emd_mode
        )
        emd_out = emd_compute(res)
        if out_path is not None:
            emd_out.to_csv(os.path.join(out_path, "res_" + f), index=False)

        # trick to add the result two times if the clustering method is None
        for cluster_method in clustering_method:
            base_dict = (
                emd_out.drop(["val_sample_ind", "steps_ahead"], axis=1)
                .mean()
                .to_dict()
            )
            base_dict.update(
                {
                    "name": f[:-4],
                    "nr_group": nr_groups,
                    "clustering": cluster_method,
                    "loss": loss_fn,
                    "station-wise MAE": res["station_wise_error"].mean(),
                }
            )
            emd_results.append(base_dict)
        print(emd_results[-1])
    emd_results = pd.DataFrame(emd_results)
    return emd_results


def make_plots_basic(results, out_path):
    basic_results = results[results["name"].str.contains("basic")]
    assert len(
        basic_results.drop_duplicates(subset=["nr_group", "clustering"])
    ) == len(basic_results)
    results["clustering"] = results["clustering"].map(
        {"kmeans": "K-Means", "agg": "agglomerative"}
    )

    for var in ["EMD", "MAE", "station-wise MAE"]:
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            basic_results, x="nr_group", y=var, hue="clustering", linewidth=3
        )
        plt.xlabel("Number of station-clusters")
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"lines_{var}.pdf"))


def loss_comparison(results, out_path):
    # first get all the ones with special loss functions
    loss_comparison = results[results["loss"] != "basic"]
    if len(loss_comparison) == 0:
        print("No loss comparison because no special losses are used")
        return 1
    loss_comparison["Method"] = (
        results["loss"]
        + "_"
        + results["nr_group"].astype(str)
        + "_"
        + results["clustering"]
    )
    evaluated_levels = loss_comparison["nr_group"].unique()
    # Secondly get the ones with basic loss at the same clustering level
    comparative_results = results[
        results["nr_group"].isin(evaluated_levels)
        & (results["loss"] == "basic")
    ]
    comparative_results["Method"] = (
        "basic_" + results["nr_group"].astype(str) + "_" + results["clustering"]
    )
    loss_comparison = pd.concat([loss_comparison, comparative_results]).dropna()

    for var in ["EMD", "MAE", "station-wise MAE"]:
        plt.figure(figsize=(10, 10))
        sns.barplot(data=loss_comparison.sort_values([var]), x="Method", y=var)
        if var == "EMD":
            plt.ylim(0, 700)
        elif var == "MAE":
            plt.ylim(0, 1.5)
        else:
            plt.ylim(0, 0.75)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"lossbar_{var}.pdf"))


def correlate_mae_emd(out_path):
    # load results
    single_station_res = get_singlestations_file(out_path)
    plt.figure(figsize=(6, 4))
    plt.scatter(single_station_res["MAE"], single_station_res["EMD"])
    plt.ylabel("EMD")
    plt.xlabel("MAE")
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "emd_vs_mae.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True)
    parser.add_argument("-p", "--path", type=str, default="outputs")
    parser.add_argument(
        "--redo", action="store_true", help="for processing the results again"
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="group_to_group",
        help="EMD mode: station_to_station / group_to_station / group_to_group",
    )
    parser.add_argument(
        "--steps_ahead",
        default=-1,
        type=int,
        help="filter for one specific step-ahead",
    )
    args = parser.parse_args()

    comp = args.name
    comp_path = os.path.join(args.path, comp)
    out_plot_name = (
        "plots" if args.steps_ahead == -1 else "plots" + str(args.steps_ahead)
    )
    out_path = os.path.join(args.path, comp + "_" + out_plot_name)
    os.makedirs(out_path, exist_ok=True)

    DATASET = get_dataset_name(comp)

    # compute normal results
    if args.redo or not os.path.exists(out_path):
        emd_results = compare_emd(
            comp_path,
            filter_step=args.steps_ahead,
            emd_mode=args.mode,
            out_path=out_path,
        )
        emd_results.to_csv(os.path.join(out_path, f"results.csv"), index=False)
    else:
        # if they already exist, load results
        emd_results = pd.read_csv(os.path.join(out_path, f"results.csv"))

    # compare losses
    loss_comparison(emd_results, out_path)
    # compare aggregation layers
    make_plots_basic(emd_results, out_path)

    correlate_mae_emd(out_path)
