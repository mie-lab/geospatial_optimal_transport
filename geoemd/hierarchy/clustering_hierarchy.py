import json
from sklearn.cluster import KMeans, AgglomerativeClustering

clustering_dict = {"kmeans": KMeans, "agg": AgglomerativeClustering}


class SpatialClustering:
    def __init__(self, stations):
        self.stations = stations

    def __call__(self, clustering_method="kmeans", n_clusters=10):
        cluster_class = clustering_dict[clustering_method](
            n_clusters=n_clusters
        )
        cluster_class.fit(self.stations[["x", "y"]])
        self.stations["cluster"] = cluster_class.labels_
        self.stations["cluster"] = "Group_" + self.stations["cluster"].astype(
            str
        )
        self.groups_coordinates = self.stations.groupby("cluster").agg(
            {"x": "mean", "y": "mean"}
        )

    def transform_demand(self, demand_df_inp, hierarchy=False):
        """demand_df: Dataframe with rows = timestamps and columns=station ids"""
        # merge with stations
        demand_df = demand_df_inp.swapaxes(1, 0)
        demand_df["cluster"] = self.stations["cluster"]
        demand_grouped = demand_df.groupby("cluster").sum()
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
