from scipy.spatial.distance import cdist
import wasserstein
import pandas as pd

from utils import get_children_hierarchy


# TODO: possibly make some objects such that hierarchy is stored permanently in the object
def transport_from_centers(groups_with_pred, base_station, hier):
    base_station_coords = base_station[["start_x", "start_y"]].values.astype(
        float
    )
    base_station_dist = base_station["dist"].values
    print("OPTION 1")
    # OPTION 1
    # for level, level_df in groups_with_preds.groupby("nr_stations"):
    for level in range(1, 12, 1):
        level_stations = get_children_hierarchy("Group_48", hier, level)
        #     print(level_stations)
        #     print(level, level_stations)
        level_df = (
            groups_with_pred.loc[level_stations].copy().drop("gt", axis=1)
        )

        level_df["dist"] = level_df["pred"] / level_df["pred"].sum()

        # compute pairwise distances between current groups and the stations
        pairwise_distances = cdist(
            level_df[["start_x", "start_y"]].values.astype(float),
            base_station_coords,
        )

        was = wasserstein.EMD()
        emd_distance = was(
            level_df["dist"].values.flatten(),
            base_station_dist,
            pairwise_distances,
        )
        print(level, emd_distance)


def transport_equal_dist(groups_with_pred, base_station, hier):
    print("OPTION 2")
    base_station_coords = base_station[["start_x", "start_y"]].values.astype(
        float
    )
    base_station_dist = base_station["dist"].values
    # use just the predictions
    pred_per_group = groups_with_pred["pred"]

    # cdist is always the distance between all stations with each other
    dist_matrix = cdist(base_station_coords, base_station_coords)

    # for each group, make one df which defines the distribution over stations (given the prediction)
    distributed_preds = {}
    for group in pred_per_group.index:
        pred_this_group = pred_per_group.loc[group]
        stations_of_group = get_children_hierarchy(group, hier, 100)
        distributed_preds[group] = pd.Series(
            pred_this_group / len(stations_of_group), index=stations_of_group
        )

    # iterate over levels
    for level in range(1, 12, 1):
        level_stations = get_children_hierarchy("Group_48", hier, level)

        # Gather all predictions per stations
        preds_per_stations_level = []
        for sta in level_stations:
            preds_per_stations_level.append(distributed_preds[sta])
        level_df = pd.DataFrame(
            pd.concat(preds_per_stations_level), columns=["pred"]
        )
        #     print(level_df["pred"].sum())
        # normalize
        level_df["dist"] = level_df["pred"] / level_df["pred"].sum()

        # reorder! -> need to align with the base station index
        level_df = level_df.loc[base_station.index]

        was = wasserstein.EMD()
        emd_distance = was(
            level_df["dist"].values.flatten(), base_station_dist, dist_matrix
        )
        print(level, emd_distance)
