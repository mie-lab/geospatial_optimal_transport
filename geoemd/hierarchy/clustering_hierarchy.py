import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from geoemd.utils import space_cost_matrix
from geoemd.hierarchy.hierarchy_utils import clustered_cost_matrix

clustering_dict = {"kmeans": KMeans, "agg": AgglomerativeClustering}


class SpatialClustering:
    def __init__(self, in_path_stations: str, is_cost_matrix: bool = False):
        # load stations or cost matrix -> both have a column station_id that we
        # will set as the index.
        self.stations = pd.read_csv(in_path_stations).set_index("station_id")
        self.stations.index = self.stations.index.astype(str)
        # indicate whether the stations are a dataframe or directly a cost matrix
        self.is_cost_matrix = is_cost_matrix

    def cluster_spectral(self, n_clusters, gauss_variance=1000):
        """Clustering from graph adjacency matrix"""
        # init spectral clustering
        clustering = SpectralClustering(n_clusters=n_clusters)
        # transform cost matrix into adjacency matrix
        adj_matrix = np.exp(
            -self.stations.values**2 / (2.0 * gauss_variance**2)
        )
        clustering.fit(adj_matrix)
        # return labels
        return clustering.labels_

    def __call__(self, clustering_method="kmeans", n_clusters=10):
        if self.is_cost_matrix:
            cluster_labels = self.cluster_spectral(n_clusters=n_clusters)
        else:
            cluster_class = clustering_dict[clustering_method](
                n_clusters=n_clusters
            )
            cluster_class.fit(self.stations[["x", "y"]])
            cluster_labels = cluster_class.labels_
        self.stations["cluster"] = cluster_labels
        self.stations["cluster"] = "Group_" + self.stations["cluster"].astype(
            str
        )

    def get_clustered_cost_matrix(self, time_series_cols: list):
        """
        Outputs a group-to-group cost matrix
        Args:
            time_series_cols (list): list of column names to get the order of
            the stations right

        Returns:
            time_dist_matrix (np.ndarry): kxk matrix with the costs between each
            pair of clusters
        """
        # case 1: we have the station coordinates
        if not self.is_cost_matrix:
            groups_coordinates = self.stations.groupby("cluster").agg(
                {"x": "mean", "y": "mean"}
            )
            # get group coordinates
            station_coords = groups_coordinates.loc[time_series_cols].values
            time_dist_matrix = space_cost_matrix(station_coords)
        # case 2: we have only the cost matrix itself
        else:
            # cluster the original cost matrix by the clusters
            # avg distance from these points to the ones from another cluster
            time_dist_matrix = clustered_cost_matrix(
                self.stations, time_series_cols
            )
        return time_dist_matrix

    def transform_demand(self, demand_df_inp, hierarchy=False, agg_func="mean"):
        """demand_df: Dataframe with rows = timestamps and columns=station ids"""
        # merge with stations
        demand_df = demand_df_inp.swapaxes(1, 0)
        demand_df.index = demand_df.index.astype(str)
        demand_df["cluster"] = self.stations["cluster"]
        if agg_func == "sum":
            demand_grouped = demand_df.groupby("cluster").sum()
        else:
            demand_grouped = demand_df.groupby("cluster").mean()
        demand_grouped = demand_grouped.swapaxes(1, 0)
        if hierarchy:
            total = demand_df_inp.sum(axis=1)
            demand_grouped = pd.concat([demand_df_inp, demand_grouped], axis=1)
            demand_grouped["total"] = total

        demand_grouped = (
            demand_grouped.reset_index()
            .rename_axis(None, axis=1)
            .set_index("timeslot")
        )
        return demand_grouped

    def get_topdown_hierarchy(self):
        station_cluster_dict = (
            self.stations.reset_index()
            .groupby("cluster")["station_id"]
            .unique()
            .to_dict()
        )
        station_cluster_dict = {
            k: [int(v_val) for v_val in v]
            for k, v in station_cluster_dict.items()
        }
        station_cluster_dict["total"] = list(self.stations["cluster"].unique())
        return station_cluster_dict

    def get_darts_hier(self):
        darts_hier = {}
        for s_id in self.stations.index:
            darts_hier[str(s_id)] = [self.stations.loc[s_id, "cluster"]]
        for cluster_id in self.stations["cluster"].unique():
            darts_hier[cluster_id] = ["total"]
        return darts_hier

    def save(self, save_path):
        with open(save_path, "w") as outfile:
            json.dump(self.get_topdown_hierarchy(), outfile)
