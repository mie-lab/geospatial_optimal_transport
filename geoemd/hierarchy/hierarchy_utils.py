import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from darts import TimeSeries
from geoemd.hierarchy.full_station_hierarchy import FullStationHierarchy
from geoemd.io import load_stations


def aggregate_bookings_deprecated(demand_df, agg_by="day"):
    if agg_by == "day":
        demand_df[agg_by] = demand_df["timeslot"].dt.date
    elif agg_by == "hour":
        demand_df[agg_by] = (
            demand_df["timeslot"].dt.date.astype(str)
            + "-"
            + demand_df["timeslot"].dt.hour.astype(str)
        )
    else:
        #     demand_df["second_hour"] = demand_df["hour"] // 2
        raise NotImplementedError()

    # count bookings per aggregatione time
    bookings_agg = demand_df.groupby([agg_by, "station_id"])[
        "duration_sec"
    ].count()
    bookings_agg = pd.DataFrame(bookings_agg).reset_index()
    bookings_agg.rename(
        {"duration_sec": "demand", agg_by: "timeslot"}, axis=1, inplace=True
    )
    bookings_agg = demand_df.pivot(
        index="timeslot", columns="station_id", values="count"
    ).fillna(0)
    return bookings_agg


def clustering_algorithm(stations_locations):
    stations_locations.sort_values("station_id")
    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    clustering.fit(stations_locations[["x", "y"]])
    return clustering.children_


def hierarchy_to_df(res_hierarchy):
    # convert to df
    station_group_df = []
    for group, stations_in_group in res_hierarchy.items():
        if group == "total":
            continue
        df = pd.DataFrame()
        if type(stations_in_group) == dict:
            df["station"] = stations_in_group.keys()
            df["station_fraction"] = stations_in_group.values()
        else:
            df["station"] = stations_in_group
        df["group"] = group
        df["nr_stations"] = len(stations_in_group)
        station_group_df.append(df)
    station_group_df = pd.concat(station_group_df).sort_values("station")
    return station_group_df


def add_fraction_to_hierarchy(res_hierarchy, train_data):
    # initialize hierachy
    hierarchy_with_fractions = {}
    for group_id, group_stations in res_hierarchy.items():
        hierarchy_with_fractions[group_id] = {}
        if group_id != "total":
            train_data[group_id] = train_data[group_stations].sum(axis=1)
            for station in group_stations:
                # TODO: the better version would be (data[station] /
                # data[group_id]).mean(), but leads to zero division
                hierarchy_with_fractions[group_id][station] = (
                    train_data[station].sum() / train_data[group_id].sum()
                )
    return hierarchy_with_fractions


def clustered_cost_matrix(
    cost_matrix: pd.DataFrame, cluster_list: list
) -> np.ndarray:
    """
    Compute cost matrix between a pair of clusters
    Args:
        cost_matrix (pd.DataFrame): dataframe with one row and one col per
            station, where the values indicate the cost between a pair
        cluster_list (list): list of cluster names

    Returns:
        np.ndarray: matrix with one row per cluster and one column per cluster
    """
    assert (
        "cluster" in cost_matrix.columns
    ), "cost_matrix requires column cluster"
    time_dist_matrix = np.zeros((len(cluster_list), len(cluster_list)))
    for i, cluster1 in enumerate(cluster_list):
        for j, cluster2 in enumerate(cluster_list):
            # only do it for i<j because i=j is 0 and others are
            # added symmetrically
            if i >= j:
                continue
            rows_cluster1 = cost_matrix[cost_matrix["cluster"] == cluster1]
            columns_cluster2 = (
                cost_matrix[cost_matrix["cluster"] == cluster2]
            ).index.astype(str)
            cost_to_other_cluster = rows_cluster1[columns_cluster2]

            # fill symmetrically with mean
            time_dist_matrix[i, j] = np.mean(cost_to_other_cluster)
            time_dist_matrix[j, i] = np.mean(cost_to_other_cluster)
    return time_dist_matrix


def group_to_station_cost_matrix(
    cost_matrix: pd.DataFrame, cluster_list: list
) -> np.ndarray:
    """
    Compute cost matrix between clusters and individual stations
    Args:
        cost_matrix (pd.DataFrame): dataframe with one row and one col per
            station, where the values indicate the cost between a pair
        cluster_list (list): list of cluster names

    Returns:
        np.ndarray: matrix with one row per cluster and one column per cluster
    """
    assert (
        "cluster" in cost_matrix.columns
    ), "cost_matrix requires column cluster"
    # group_to_station -> between cluster and single stations = len(cost_matrix)
    time_dist_matrix = np.zeros((len(cluster_list), len(cost_matrix)))

    for i, cluster in enumerate(cluster_list):
        for j, station in enumerate(cost_matrix.index):
            # only do it for i<j because i=j is 0 and others are
            # added symmetrically
            if i >= j:
                continue
            rows_cluster1 = cost_matrix[cost_matrix["cluster"] == cluster]
            cost_to_station = rows_cluster1[station]

            # fill symmetrically with mean
            time_dist_matrix[i, j] = np.mean(cost_to_station)
            time_dist_matrix[j, i] = np.mean(cost_to_station)
    return time_dist_matrix


# # Deprecated
# def demand_hierarchy(bookings_agg, linkage, nr_samples=len(stations_locations)):
#     # initialize hierarchy
#     hierarchy = np.zeros((len(linkage) + nr_samples, nr_samples))
#     hierarchy[:nr_samples] = np.identity(nr_samples)

#     for i, pair in enumerate(linkage):
#         bookings_agg[i + nr_samples] = (
#             bookings_agg[pair[0]] + bookings_agg[pair[1]]
#         )
#         # add to hierarchy
#         row_for_child1 = hierarchy[pair[0]]
#         row_for_child2 = hierarchy[pair[1]]
#         hierarchy[i + nr_samples] = np.logical_or(
#             row_for_child1, row_for_child2
#         )

#     # convert to string columns
#     bookings_agg = (
#         bookings_agg.reset_index()
#         .rename_axis(None, axis=1)
#         .set_index("timeslot")
#     )
#     bookings_agg.columns = bookings_agg.columns.astype(str)

#     return bookings_agg, hierarchy


def hierarchy_to_dict(linkage, nr_samples):
    hier = {}
    for i, pair in enumerate(linkage):
        hier[str(i + nr_samples)] = list(pair.astype(str))
    return hier


def test_hierarchy(bookings_agg, hierarchy, test_node=800):
    # only works if only two
    #     if len(np.where(hierarchy[test_node])[0])==2:
    #         assert np.all(np.where(hierarchy[test_node]) == linkage[test_node - nr_samples])

    # assert that the column relations correspond to the hierarchy
    inds = np.where(hierarchy[test_node])[0]
    summed = bookings_agg[inds[0]].copy()
    for k in inds[1:]:
        summed += bookings_agg[k]
    assert all(bookings_agg[test_node] == summed)


def add_demand_groups(demand_agg, hier):
    """Add groups of station to time series dataframe"""
    demand_agg.columns = demand_agg.columns.astype(str)
    demand_agg.index = pd.to_datetime(demand_agg.index)
    for key, pair in hier.items():
        demand_agg[key] = (
            demand_agg[pair[0]].values + demand_agg[pair[1]].values
        )
    return demand_agg


def construct_series_with_hierarchy(
    demand_agg: pd.DataFrame, in_path_stations: str, frequency: str
):
    stations_locations = load_stations(in_path_stations)
    station_hierarchy = FullStationHierarchy()
    if "0" in demand_agg.columns:
        demand_agg.drop("0", axis=1, inplace=True)
        stations_locations = stations_locations[stations_locations.index != 0]
    station_hierarchy.init_from_station_locations(stations_locations)
    demand_agg = add_demand_groups(demand_agg, station_hierarchy.hier)
    # initialize time series with hierarchy
    main_time_series = TimeSeries.from_dataframe(
        demand_agg,
        freq=frequency,
        hierarchy=station_hierarchy.get_darts_hier(),
        fillna_value=0,
    )
    return main_time_series
