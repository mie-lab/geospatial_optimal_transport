from scipy.stats import spearmanr
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import wasserstein
from geoemd.loss.sinkhorn_loss import sinkhorn_loss_from_numpy


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


if __name__ == "__main__":
    import pandas as pd

    in_path = "outputs/comp_17_05_all/"
    model_path = "0_24_1_nhits_multi_50_3_3_0.csv"
    res_gt = pd.read_csv(in_path + "gt.csv")
    res_pred = pd.read_csv(in_path + model_path)
    station_groups = pd.read_csv("../data/bikes_montreal/test_stations.csv")
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
