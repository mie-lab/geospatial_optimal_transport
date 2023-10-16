import wasserstein
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import ot
import torch

from geoemd.hierarchy.hierarchy_utils import hierarchy_to_df
from geoemd.utils import space_cost_matrix
from geoemd.loss.sinkhorn_loss import SinkhornLoss
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
    def __init__(self, stations, gt_reference):
        self.gt_reference = gt_reference.set_index(
            ["val_sample_ind", "steps_ahead"]
        ).rename({"group": "station"}, axis=1)
        self.stations = stations
        assert self.stations.index.name == "station_id"
        self.stations.sort_index(inplace=True)
        # default dist matrix: between stations
        self.dist_matrix = space_cost_matrix(
            self.stations[["x", "y"]], quadratic=True, speed_factor=10
        )

    def __call__(self, res, res_hierarchy, mode="station_to_station"):
        if mode == "station_to_station":
            return self.emd_station_to_station(res, res_hierarchy)
        elif mode == "group_to_group":
            return self.emd_group_to_group(res, res_hierarchy)
        elif mode == "group_to_station":
            return self.emd_group_to_station(res, res_hierarchy)
        else:
            raise ValueError("Invalid mode")

    def emd_station_to_station(
        self, res: pd.DataFrame, res_hierarchy: dict = None
    ) -> list:
        if res_hierarchy is None:
            # only possibility is station-to-station mode,
            # because there are no groups
            res_per_station = res.copy()
            res_per_station["pred_emd"] = res_per_station["pred"]
            res_per_station["station"] = res_per_station["group"]
        else:
            station_group_df = hierarchy_to_df(res_hierarchy)
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

    def emd_group_to_station(
        self, res: pd.DataFrame, res_hierarchy: dict
    ) -> list:
        if res_hierarchy is None:
            return self.emd_station_to_station(res, res_hierarchy)
        coords_per_group = self.get_coords_per_group(res_hierarchy)
        # distance matrix is between groups (pred) and stations (gt)
        self.dist_matrix = space_cost_matrix(
            coords_per_group[["x", "y"]], self.stations[["x", "y"]]
        )
        # predictions are just the per-group predictions
        res["pred_emd"] = res["pred"].copy()
        res["station"] = res["group"]
        # compute EMD
        return self.compute_emd(res)

    def emd_group_to_group(self, res: pd.DataFrame, res_hierarchy: dict):
        if res_hierarchy is None:
            return self.emd_station_to_station(res, res_hierarchy)
        coords_per_group = self.get_coords_per_group(res_hierarchy)
        # distance matrix is between groups
        self.dist_matrix = space_cost_matrix(coords_per_group[["x", "y"]])
        # the prediction is just the per group prediction
        res["pred_emd"] = res["pred"]
        res["station"] = res["group"]
        # the gt is in the res
        self.gt_reference = res.set_index(["val_sample_ind", "steps_ahead"])

        return self.compute_emd(res)

    def compute_emd(self, res_per_station: pd.DataFrame) -> list:
        # compute emd
        was = wasserstein.EMD()
        normed_dist_matrix = self.dist_matrix / np.max(self.dist_matrix)
        sinkhorn = SinkhornLoss(self.dist_matrix)
        emd = []
        for (val_sample, steps_ahead), sample_df in res_per_station.groupby(
            ["val_sample_ind", "steps_ahead"]
        ):
            sample_df = sample_df.sort_values("station")
            pred_vals = sample_df["pred_emd"] / sample_df["pred_emd"].sum()
            # get corresponding ground truth df
            gt_df = self.gt_reference.loc[val_sample, steps_ahead].sort_values(
                "station"
            )
            gt_vals = gt_df["gt"] / gt_df["gt"].sum()
            # compute wasserstein (wo entropy)
            emd_distance = was(
                pred_vals,
                gt_vals,
                self.dist_matrix,
            )
            # compute unbalanced OT with POT package
            unbalanced_ot = ot.sinkhorn_unbalanced2(
                sample_df["pred_emd"].values,
                sample_df["gt"].values,
                normed_dist_matrix,
                0.01,
                10,
                method="sinkhorn",
                numItermax=200,
                stopThr=1e-06,
                verbose=False,
                log=False,
            ).item()
            # compute sinkhorn loss with geomloss package
            sinkhorn_loss = sinkhorn(
                torch.tensor([sample_df["pred_emd"].values.tolist()]),
                torch.tensor([gt_df["gt"].values.tolist()]),
            )
            emd.append(
                {
                    "Wasserstein": emd_distance,
                    "OT unbalanced": unbalanced_ot,
                    "sinkhorn": sinkhorn_loss.item() * 1000,
                    "val_sample_ind": val_sample,
                    "steps_ahead": steps_ahead,
                }
            )

        return pd.DataFrame(emd)
