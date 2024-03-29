import os
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import wasserstein
from scipy.special import softmax
import torch

from geoemd.loss.sinkhorn_loss import sinkhorn_loss_from_numpy
from geoemd.loss.moransi import MoransiLoss
from geoemd.emd_eval import EMDWrapper


class EMDCalibrator(EMDWrapper):
    def compute_emd(self, res_per_station, scale_factor=8):
        # # for testing moran's I
        # morans_obj = MoransiLoss(self.dist_matrix)
        torch_res, emd_res, mae_res = [], [], []
        for (val_sample, steps_ahead), sample_df in res_per_station.groupby(
            ["val_sample_ind", "steps_ahead"]
        ):
            sample_df = sample_df.sort_values("station")
            # get corresponding ground truth df
            gt_df = self.gt_reference.loc[val_sample, steps_ahead].sort_values(
                "station"
            )

            # normalize the values as they are normalized in the main training
            pred_quantile_scaled = sample_df["pred_emd"].values / scale_factor
            gt_quantile_scaled = gt_df["gt"].values / scale_factor

            # compute sinkhorn loss
            torch_res.append(
                sinkhorn_loss_from_numpy(
                    np.expand_dims(pred_quantile_scaled, 0),
                    np.expand_dims(gt_quantile_scaled, 0),
                    self.dist_matrix,
                    mode="balancedSoftmax",
                )
            )

            # compute MAE
            mae_res.append(
                np.mean(np.abs(pred_quantile_scaled - gt_quantile_scaled))
            )

            # normal normalization is necessary for normal evaluation
            pred_vals = sample_df["pred_emd"] / sample_df["pred_emd"].sum()
            gt_vals = gt_df["gt"] / gt_df["gt"].sum()
            # compute wasserstein
            was = wasserstein.EMD()
            emd_res.append(
                was(
                    pred_vals,
                    gt_vals,
                    self.dist_matrix,
                )
            )
            # # For testing Moran's I
            # torch_res.append(
            #     morans_obj(
            #         torch.from_numpy(np.expand_dims(pred_vals.values, 0)),
            #         torch.from_numpy(np.expand_dims(gt_vals.values, 0)),
            #     )
            # )
        print("Spearman", round(spearmanr(torch_res, emd_res)[0], 4))
        print("Spearman with MAE", round(spearmanr(emd_res, mae_res)[0], 4))
        return torch_res, emd_res


def compare_was(res, iters=300, sinkhorn_kwargs={}):
    torch_res, emd_res = [], []
    for _ in range(iters):
        n_samples = 5  # np.random.randint(5, 40, 1)
        rand_rows = res.sample(n_samples)
        a = np.clip(rand_rows["pred"].values, 0, None)
        if np.all(a == 0):
            a = np.ones(len(a))
        b = rand_rows["gt"].values
        if np.all(b == 0):
            b = np.ones(len(b))
        test_cdist = rand_rows[["x", "y"]].values
        test_cdist = cdist(test_cdist, test_cdist)

        # 1) sinkhorn
        torch_res.append(
            sinkhorn_loss_from_numpy(
                np.expand_dims(a, 0),
                np.expand_dims(b, 0),
                test_cdist,
                sinkhorn_kwargs,
            )
        )
        # 2) real wasserstein distance
        # normalize to values between 0 and 1
        test_cdist = test_cdist / np.max(test_cdist)
        a = a / np.sum(a)
        b = b / np.sum(b)

        was = wasserstein.EMD()
        emd_res.append(was(a, b, test_cdist))

    print("Spearman", round(spearmanr(torch_res, emd_res)[0], 4))
    plt.scatter(torch_res, emd_res)
    #     plt.xscale("log")
    #     plt.yscale("log")
    plt.show()


def check_pred_gt(res, sinkhorn_kwargs={}):
    res_stations = res[~res.index.str.contains("Group")]

    test_cdist = res_stations[["x", "y"]].values
    test_cdist = cdist(test_cdist, test_cdist)
    test_cdist_normed = test_cdist / np.max(test_cdist)

    torch_res, emd_res = [], []
    for i in range(5):
        gt_vals = res_stations["gt_" + str(i)].values
        gt_vals = gt_vals / np.sum(gt_vals)

        model_pred_cols = [
            c for c in res.columns if "_" + str(i) in c and c.startswith("pred")
        ]

        for model_pred in model_pred_cols:
            a = res_stations[model_pred].values
            a = a / np.sum(a)

            torch_res.append(
                sinkhorn_loss_from_numpy(
                    a, gt_vals, test_cdist_normed, sinkhorn_kwargs
                )
            )
            was = wasserstein.EMD()
            emd_res.append(was(a, gt_vals, test_cdist_normed))
    #     emd_unnormed.append(was(a, gt_vals, test_cdist))


def old_main_wasserstein_calibration():
    in_path = "outputs/comp_17_05_all/"
    model_path = "0_24_1_nhits_multi_50_3_3_0.csv"
    res_gt = pd.read_csv(in_path + "gt.csv")
    res_pred = pd.read_csv(in_path + model_path)
    station_groups = pd.read_csv("data/bikes/test_stations.csv")
    together = res_pred.merge(
        res_gt,
        left_on=["group", "steps_ahead", "val_sample_ind"],
        right_on=["group", "steps_ahead", "val_sample_ind"],
        how="left",
    )
    together = together.merge(
        station_groups, how="left", left_on="group", right_on="station_id"
    )
    compare_was(together)


def load_stations(station_path="data/bikes/test_stations.csv"):
    return (
        pd.read_csv(station_path)
        .sort_values("station_id")
        .set_index("station_id")
    )


if __name__ == "__main__":
    comp = "bikes_cluster_17_06"
    path = os.path.join("outputs", comp)
    out_path = os.path.join("outputs", comp + "plots")

    gt_file = [f for f in os.listdir(path) if "None" in f][0]
    # load gt as reference (per station gt needed for evaluation)
    single_station_res = pd.read_csv(os.path.join(path, gt_file))

    stations = load_stations()
    single_station_res = single_station_res[
        single_station_res["steps_ahead"] == 0
    ]

    calib = EMDCalibrator(
        stations,
        single_station_res.drop("pred", axis=1),
        mode="station_to_station"
    )
    torch_res, emd_res = calib(single_station_res)
    plt.scatter(emd_res, torch_res)
    plt.show()
