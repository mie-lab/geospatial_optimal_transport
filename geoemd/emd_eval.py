import wasserstein
import time
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import ot
import torch

from geoemd.hierarchy.hierarchy_utils import hierarchy_to_df
from geoemd.utils import space_cost_matrix
from geoemd.loss.sinkhorn_loss import SinkhornLoss
from geoemd.loss.interpretable_unbalanced_ot import InterpretableUnbalancedOT
from geoemd.config import SPEED_FACTOR

"""
MODES:
station_to_station: We have a prediction per group,
but split it over the included stations in equal parts,
then compute emd with cdist station-station
group_to_station: We have a prediction per group, we just
take the center coordinates of the group and use the
distances between group centers and stations as cdist
group_to_group: We just compute the EMD error in terms of
rebalancing between the clusters
"""


class EMDWrapper:
    def __init__(
        self,
        stations,
        gt_reference,
        res_hierarchy=None,
        quadratic_cost=False,
        mode="station_to_station",
    ):
        self.mode = mode
        self.gt_reference = gt_reference.set_index(
            ["val_sample_ind", "steps_ahead"]
        ).rename({"group": "station"}, axis=1)
        self.stations = stations
        self.res_hierarchy = res_hierarchy
        assert self.stations.index.name == "station_id"
        self.stations.sort_index(inplace=True)
        # if no hierarchy, we have only station-wise data
        if res_hierarchy is None:
            self.mode = "station_to_station"
        else:
            coords_per_group = self.get_coords_per_group(res_hierarchy)

        # make dist matrix --> TODO: load it directly for traffic data
        if self.mode == "station_to_station":
            self.dist_matrix = space_cost_matrix(
                self.stations[["x", "y"]], quadratic=quadratic_cost
            )
        elif self.mode == "group_to_station":
            # distance matrix is between groups (pred) and stations (gt)
            self.dist_matrix = space_cost_matrix(
                coords_per_group[["x", "y"]], self.stations[["x", "y"]]
            )
        elif self.mode == "group_to_group":
            # distance matrix is between groups
            self.dist_matrix = space_cost_matrix(coords_per_group[["x", "y"]])
        else:
            raise ValueError("Invalid mode")

    def __call__(self, res):
        if self.mode == "station_to_station":
            return self.emd_station_to_station(res)
        elif self.mode == "group_to_group":
            return self.emd_group_to_group(res)
        elif self.mode == "group_to_station":
            return self.emd_group_to_station(res)
        else:
            raise ValueError("Invalid mode")

    def emd_station_to_station(self, res: pd.DataFrame) -> list:
        if self.res_hierarchy is None:
            # only possibility is station-to-station mode,
            # because there are no groups
            res_per_station = res.copy()
            res_per_station["pred_emd"] = res_per_station["pred"]
            res_per_station["station"] = res_per_station["group"]
        else:
            station_group_df = hierarchy_to_df(self.res_hierarchy)
            # merge them in order to get predictions per station
            res_per_station = pd.merge(
                station_group_df,
                res,
                how="right",
                left_on="group",
                right_on="group",
            )
            # if we have the information what fraction each station has, use it
            if "station_fraction" in station_group_df.columns:
                res_per_station["pred_emd"] = (
                    res_per_station["pred"]
                    * res_per_station["station_fraction"]
                )
            else:
                # otherwise, just take the same fraction for every station
                res_per_station["pred_emd"] = (
                    res_per_station["pred"] / res_per_station["nr_stations"]
                )
        # compute EMD
        return self.compute_emd(res_per_station)

    def get_coords_per_group(self, res_hierarchy: dict) -> list:
        # get groups
        station_group_df = hierarchy_to_df(res_hierarchy)
        coords_per_group = self.stations.merge(
            station_group_df, left_index=True, how="left", right_on="station"
        )
        coords_per_group = coords_per_group.groupby("group").agg(
            {"x": "mean", "y": "mean"}
        )
        coords_per_group.sort_index(inplace=True)
        return coords_per_group

    def emd_group_to_station(self, res: pd.DataFrame) -> list:
        # predictions are just the per-group predictions
        res["pred_emd"] = res["pred"].copy()
        res["station"] = res["group"]
        # compute EMD
        return self.compute_emd(res)

    def emd_group_to_group(self, res: pd.DataFrame):
        # the prediction is just the per group prediction
        res["pred_emd"] = res["pred"]
        res["station"] = res["group"]
        # the gt is in the res
        self.gt_reference = res.set_index(["val_sample_ind", "steps_ahead"])

        return self.compute_emd(res)

    def compute_emd(self, res_per_station: pd.DataFrame) -> list:
        # compute entropy-regularized unbalanced OT (loss function)
        sinkhorn = SinkhornLoss(
            self.dist_matrix,
            blur=0.1,
            reach=0.01,
            scaling=0.1,
            mode="unbalanced",
        )
        # Wasserstein distance -> also computed with normalized costs
        was = wasserstein.EMD()
        dist_matrix_normed = self.dist_matrix / np.max(self.dist_matrix)

        # Unbalanced OT is computed with 0.1 quantile and with max cost
        unb_ot_01quantile = InterpretableUnbalancedOT(
            self.dist_matrix,
            compute_exact=True,
            normalize_c=True,
            penalty_unb=np.quantile(self.dist_matrix, 0.1),
        )
        unb_ot_max = InterpretableUnbalancedOT(
            self.dist_matrix,
            compute_exact=True,
            normalize_c=True,
            penalty_unb=np.max(self.dist_matrix),
        )

        emd = []
        for (val_sample, steps_ahead), sample_df in res_per_station.groupby(
            ["val_sample_ind", "steps_ahead"]
        ):
            # get values for this sample and this step ahead
            sample_df = sample_df.sort_values("station")
            pred_vals = sample_df["pred_emd"].values
            # get corresponding ground truth df
            gt_df = self.gt_reference.loc[val_sample, steps_ahead].sort_values(
                "station"
            )
            gt_vals = gt_df["gt"].values
            # normalize pred by aligning to gt vals
            pred_vals_normed = pred_vals / np.sum(pred_vals) * np.sum(gt_vals)

            # compute base error metrics
            mae = (sample_df["pred"] - sample_df["gt"]).abs()
            mse = mae**2

            # 1) compute wasserstein (wo entropy regularization, normalized C
            emd_distance = was(
                pred_vals_normed,
                gt_vals,
                dist_matrix_normed,
            )

            # 2) compute total error
            total_error = np.abs(np.sum(gt_vals) - np.sum(pred_vals))

            # 3) compute sinkhorn loss with geomloss package
            gt_tensor = torch.from_numpy(gt_vals).unsqueeze(0)
            pred_tensor = torch.from_numpy(pred_vals).unsqueeze(0)
            sinkhorn_loss = sinkhorn(pred_tensor, gt_tensor)

            # 4) unbalanced ot
            unb_ot_01quantile_res = unb_ot_01quantile(pred_tensor, gt_tensor)
            unb_ot_max_res = unb_ot_max(pred_tensor, gt_tensor)

            emd.append(
                {
                    "EMD": emd_distance,
                    "Unb_OT_01quantile": unb_ot_01quantile_res,
                    "Unb_OT_max": unb_ot_max_res,
                    "total_error": total_error,
                    "MAE": np.mean(mae),
                    "MSE": np.mean(mse),
                    "Sinkhorn": sinkhorn_loss.item() * 10,
                    "val_sample_ind": val_sample,
                    "steps_ahead": steps_ahead,
                }
            )

        return pd.DataFrame(emd)
