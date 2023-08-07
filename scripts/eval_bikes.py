import os
import json
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15})

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
    path, filter_step=-1, emd_mode="station_to_station", split_by_fraction=True
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
        data = data.pivot(
            index="timeslot", columns="station_id", values="count"
        ).fillna(0)
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
        clustering_method = f[:-4].split("_")[-1]
        nr_groups = int(f.split("_")[-2])
        if clustering_method == "None":
            clustering_method = "agg"
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
        emd_compute = EMDWrapper(stations, gt_reference)
        emd = emd_compute(res, res_hierarchy, mode=emd_mode)["EMD"].values

        emd_results.append(
            {
                "name": f[:-4],
                "nr_group": nr_groups,
                "clustering": clustering_method,
                "loss": loss_fn,
                "EMD": np.mean(emd),
                "EMD_std": np.std(emd),
                "MAE": res["error"].mean(),
                "MAE std": res["error"].std(),
                "station-wise MAE": res["station_wise_error"].mean(),
                "station-wise MAE std": res["station_wise_error"].std(),
            }
        )
        print(emd_results[-1])
    emd_results = pd.DataFrame(emd_results)
    return emd_results


def make_plots_basic(results, out_path):
    basic_results = results[results["name"].str.contains("basic")]
    assert len(
        basic_results.drop_duplicates(subset=["nr_group", "clustering"])
    ) == len(basic_results)

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


def correlate_mae_emd(single_station_res, out_path):
    stations = load_stations(DATASET)
    # ger error
    single_station_res["MAE"] = (
        single_station_res["gt"] - single_station_res["pred"]
    ).abs()
    mae_per_sample = pd.DataFrame(
        single_station_res.groupby(["val_sample_ind", "steps_ahead"])[
            "MAE"
        ].mean()
    )
    # get emd
    emdwrap = EMDWrapper(stations, single_station_res.drop("pred", axis=1))
    emd = emdwrap(single_station_res, None, mode="station_to_station")
    # join
    together = pd.merge(
        mae_per_sample,
        emd,
        left_index=True,
        right_on=["val_sample_ind", "steps_ahead"],
    )
    plt.figure(figsize=(6, 6))
    plt.scatter(together["MAE"], together["EMD"])
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
        "--steps",
        action="store_true",
        help="process also step 0 1 2 separately",
    )
    args = parser.parse_args()

    comp = args.name
    comp_path = os.path.join(args.path, comp)
    out_path = os.path.join(args.path, comp + "plots")
    os.makedirs(out_path, exist_ok=True)

    DATASET = get_dataset_name(comp)

    # compute normal results
    if args.redo:
        emd_results = compare_emd(comp_path)
        emd_results.to_csv(os.path.join(out_path, f"results.csv"), index=False)
    else:
        emd_results = pd.read_csv(os.path.join(out_path, f"results.csv"))

    # compare losses
    loss_comparison(emd_results, out_path)
    # compare aggregation layers
    make_plots_basic(emd_results, out_path)

    single_station_res = get_singlestations_file(comp_path)
    correlate_mae_emd(single_station_res, out_path)

    # distinguish by steps ahead:
    if args.steps:
        for steps_ahead in range(3):
            emd_results = compare_emd(
                os.path.join(args.path, comp), filter_step=steps_ahead
            )
            emd_results.to_csv(
                os.path.join(out_path, f"results_step{steps_ahead}.csv"),
                index=False,
            )
