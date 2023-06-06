import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering


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
        df["station"] = stations_in_group
        df["group"] = group
        df["nr_stations"] = len(stations_in_group)
        station_group_df.append(df)
    station_group_df = pd.concat(station_group_df).sort_values("station")
    return station_group_df


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


def cluster_agglomerative(station_locations):
    # cluster the stations
    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    clustering.fit(station_locations[["x", "y"]])
    return clustering.children_


def add_demand_groups(demand_agg, hier):
    """Add groups of station to time series dataframe"""
    demand_agg.columns = demand_agg.columns.astype(str)
    demand_agg.index = pd.to_datetime(demand_agg.index)
    for key, pair in hier.items():
        demand_agg[key] = (
            demand_agg[pair[0]].values + demand_agg[pair[1]].values
        )
    return demand_agg
