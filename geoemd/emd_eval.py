import wasserstein
import time
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import ot
import torch

from geoemd.hierarchy.hierarchy_utils import (
    hierarchy_to_df,
    clustered_cost_matrix,
    group_to_station_cost_matrix,
)
from geoemd.utils import space_cost_matrix
from geoemd.loss.sinkhorn_loss import SinkhornLoss
from geoemd.loss.partial_ot import InterpretableUnbalancedOT
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

        # Case 1: stations have x and y coordinates -> make matrix with coords
        if "x" in self.stations and "y" in self.stations:
            self.make_dist_matrix_coords(quadratic_cost)
        else:
            self.make_dist_matrix_adjacency()

    def make_dist_matrix_adjacency(self) -> None:
        """Construct dist matrix from adjacency matrix"""
        if self.mode == "station_to_station" or self.res_hierarchy is None:
            self.dist_matrix = self.stations.values
        else:
            station_group_df = (
                hierarchy_to_df(self.res_hierarchy)
                .rename({"group": "cluster", "station": "station_id"}, axis=1)
                .set_index("station_id")
            )
            self.stations = self.stations.merge(
                station_group_df[["cluster"]],
                left_index=True,
                right_index=True,
                how="left",
            )
            sorted_group_list = sorted(self.stations["cluster"].unique())
            if self.mode == "group_to_station":
                # compute average distances from cluster to individual stations
                self.dist_matrix = group_to_station_cost_matrix(
                    self.stations, sorted_group_list
                )
            elif self.mode == "group_to_group":
                # compute average distance between clusters
                self.dist_matrix = clustered_cost_matrix(
                    self.stations, sorted_group_list
                )
            else:
                raise ValueError("Invalid mode")

    def make_dist_matrix_coords(self, quadratic_cost: bool) -> None:
        """Construct dist matrix from x y station coordinates"""
        if self.res_hierarchy is None:
            self.mode = "station_to_station"
        else:
            coords_per_group = self.get_coords_per_group()

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

    def get_coords_per_group(self) -> list:
        # get groups
        station_group_df = hierarchy_to_df(self.res_hierarchy)
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
        # dist_matrix_normed = self.dist_matrix / np.max(self.dist_matrix)

        # Unbalanced OT is computed with different quantiles
        partial_ot_objects = []
        quantile_range = [0] + np.arange(0.1, 1.1, 0.1).tolist() +  [1.5 * np.max(self.dist_matrix)]
        names = ["quantile_"+str(i) for i in range(11)] + ["1.5max"]
        penalty_mapping = {}
        for i in range(len(quantile_range)):
            if "quantile" in names[i]:
                penalty = np.quantile(self.dist_matrix, quantile_range[i])
            else:
                penalty = quantile_range[i]
            penalty_mapping[names[i]] = penalty
            # print(names[i], penalty)
            partial_ot_objects.append(InterpretableUnbalancedOT(
            self.dist_matrix,
            compute_exact=True,
            normalize_c=False,
            penalty_unb=penalty,
        ))
        print(penalty_mapping)

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
            # pred_vals_normed = pred_vals / np.sum(pred_vals) # * np.sum(gt_vals)
            gt_vals_normed = gt_vals / np.sum(gt_vals) * np.sum(pred_vals)

            # compute base error metrics
            mae = (sample_df["pred"] - sample_df["gt"]).abs()
            mse = mae**2

            # 1) compute wasserstein (wo entropy regularization, normalized C
            emd_distance = was(
                pred_vals,
                gt_vals_normed,
                self.dist_matrix,
            )

            # 2) compute total error
            total_error = np.abs(np.sum(gt_vals) - np.sum(pred_vals))

            # 3) compute sinkhorn loss with geomloss package
            gt_tensor = torch.from_numpy(gt_vals).unsqueeze(0)
            pred_tensor = torch.from_numpy(pred_vals).unsqueeze(0)
            sinkhorn_loss = sinkhorn(pred_tensor, gt_tensor)

            base_dict =  {
                    "EMD": emd_distance,
                    "total_error": total_error,
                    "MAE": np.mean(mae),
                    "MSE": np.mean(mse),
                    "Sinkhorn": sinkhorn_loss.item() * 10,
                    "val_sample_ind": val_sample,
                    "steps_ahead": steps_ahead,
                }
            
            # 4) unbalanced ot
            for i in range(len(names)):
                unb_ot_res = partial_ot_objects[i](pred_tensor, gt_tensor)
                base_dict[f"unb_ot_{names[i]}"] = unb_ot_res
            emd.append(base_dict)

        return pd.DataFrame(emd)
