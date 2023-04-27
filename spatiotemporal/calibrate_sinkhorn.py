from scipy.stats import spearmanr
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import wasserstein
from sinkhorn_loss import sinkhorn_loss_from_numpy


def compare_was(res, iters=300, sinkhorn_kwargs={}):
    torch_res, emd_res = [], []
    for _ in range(iters):
        n_samples = np.random.randint(5, 40, 1)
        rand_rows = res.sample(n_samples)
        a = rand_rows["pred_linear_multi_no_0"].values
        b = rand_rows["gt_0"].values
        test_cdist = rand_rows[["x", "y"]].values
        test_cdist = cdist(test_cdist, test_cdist)
        # normalize to values between 0 and 1
        test_cdist = test_cdist / np.max(test_cdist)
        a = a / np.sum(a)
        b = b / np.sum(b)

        torch_res.append(
            sinkhorn_loss_from_numpy(a, b, test_cdist, sinkhorn_kwargs)
        )
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

    res = pd.read_csv("outputs/test1/station_groups.csv", index_col="group")
    compare_was(res)
