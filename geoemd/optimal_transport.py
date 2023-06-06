from scipy.spatial.distance import cdist
import wasserstein
import pandas as pd

from utils import get_children_hierarchy
from geoemd.hierarchy.full_station_hierarchy import FullStationHierarchy


class OptimalTransportLoss:
    def __init__(self, station_hierarchy: FullStationHierarchy):
        self.station_hierarchy = station_hierarchy

    def transport_from_centers(self, gt_col, pred_col):
        # create base stations
        base_station = self.station_hierarchy.create_base_station(gt_col)
        base_station_coords = base_station[["x", "y"]].values.astype(float)
        base_station_dist = base_station["dist"].values

        print("OPTION 1")
        # OPTION 1
        # for level, level_df in groups_with_preds.groupby("nr_stations"):
        for level in range(1, 12, 1):
            level_stations = get_children_hierarchy(
                "Group_48", self.station_hierarchy.hier, level
            )
            #     print(level_stations)
            #     print(level, level_stations)
            level_df = self.station_hierarchy.station_groups.loc[
                level_stations, ["x", "y", pred_col]
            ]

            level_df["dist"] = level_df[pred_col] / level_df[pred_col].sum()

            # compute pairwise distances between groups and the stations
            pairwise_distances = cdist(
                level_df[["x", "y"]].values.astype(float),
                base_station_coords,
            )

            was = wasserstein.EMD()
            emd_distance = was(
                level_df["dist"].values.flatten(),
                base_station_dist,
                pairwise_distances,
            )
            print(level, emd_distance)

    def transport_equal_dist(self, gt_col, pred_col):
        base_station = self.station_hierarchy.create_base_station(gt_col)
        base_station_coords = base_station[["x", "y"]].values.astype(float)
        base_station_dist = base_station["dist"].values

        print("OPTION 2")
        # use just the predictions
        pred_per_group = self.station_hierarchy.station_groups[pred_col]

        # cdist is always the distance between all stations with each other
        dist_matrix = cdist(base_station_coords, base_station_coords)

        # for each group, make one df which defines the distribution over
        # stations (given the prediction)
        distributed_preds = {}
        for group in pred_per_group.index:
            pred_this_group = pred_per_group.loc[group]
            stations_of_group = get_children_hierarchy(
                group, self.station_hierarchy.hier, 100
            )
            distributed_preds[group] = pd.Series(
                pred_this_group / len(stations_of_group),
                index=stations_of_group,
            )

        # iterate over levels
        for level in range(1, 12, 1):
            level_stations = get_children_hierarchy(
                "Group_48", self.station_hierarchy.hier, level
            )

            # Gather all predictions per stations
            preds_per_stations_level = []
            for sta in level_stations:
                preds_per_stations_level.append(distributed_preds[sta])
            level_df = pd.DataFrame(
                pd.concat(preds_per_stations_level), columns=[pred_col]
            )
            # normalize
            level_df["dist"] = level_df[pred_col] / level_df[pred_col].sum()

            # reorder! -> need to align with the base station index
            level_df = level_df.loc[base_station.index]

            was = wasserstein.EMD()
            emd_distance = was(
                level_df["dist"].values.flatten(),
                base_station_dist,
                dist_matrix,
            )
            print(level, emd_distance)
