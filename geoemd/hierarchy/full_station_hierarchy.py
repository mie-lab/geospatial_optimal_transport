import os
import json
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def cluster_agglomerative(station_locations):
    # cluster the stations
    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    clustering.fit(station_locations[["x", "y"]])
    return clustering.children_


class FullStationHierarchy:
    def __init__(self):
        print("Init object either from file or from stations_locations")

    def init_from_file(self, load_path):
        self.station_groups = pd.read_csv(
            os.path.join(load_path, "station_groups.csv"), index_col="group"
        )
        with open(
            os.path.join(load_path, "station_hierarchy.json"), "r"
        ) as infile:
            self.hier = json.load(infile)

    def init_from_station_locations(
        self, stations_locations, clustering_method=cluster_agglomerative
    ):
        assert stations_locations.index.name == "station_id"

        linkage = clustering_method(stations_locations)

        # convert into dictionary as a basis for the new station-group df
        station_groups = stations_locations.reset_index()
        station_groups["nr_stations"] = 1
        station_groups["group"] = (
            station_groups["station_id"].astype(int).astype(str)
        )
        station_groups_dict = (
            station_groups.drop(["station_id"], axis=1).swapaxes(1, 0).to_dict()
        )

        nr_samples = len(stations_locations)

        hier = {}

        for j, pair in enumerate(linkage):
            node1, node2 = (
                station_groups_dict[pair[0]],
                station_groups_dict[pair[1]],
            )

            # init new node
            new_node = {}
            # compute running average of the coordinates
            new_nr_stations = node1["nr_stations"] + node2["nr_stations"]
            new_node["nr_stations"] = new_nr_stations
            new_node["x"] = (
                node1["x"] * node1["nr_stations"]
                + node2["x"] * node2["nr_stations"]
            ) / new_nr_stations
            new_node["y"] = (
                node1["y"] * node1["nr_stations"]
                + node2["y"] * node2["nr_stations"]
            ) / new_nr_stations

            # add group name
            new_node["group"] = "Group_" + str(j)

            # add to overall dictionary
            station_groups_dict[j + nr_samples] = new_node

            # add to darts hierarchy
            #         darts_hier[node1["group"]] = new_node["group"]
            #         darts_hier[node2["group"]] = new_node["group"]
            hier[new_node["group"]] = [node1["group"], node2["group"]]

        station_groups = (
            pd.DataFrame(station_groups_dict).swapaxes(1, 0).set_index("group")
        )
        self.station_groups = station_groups
        self.hier = hier

    def get_darts_hier(self):
        darts_hier = {}
        for key, pair in self.hier.items():
            for p in pair:
                darts_hier[p] = key
        return darts_hier

    def deprecated_add_pred(self, pred_xa_col, col_name):
        """
        Add a prediction from darts as a column to this df
        Note: This function is not used at the moment
        """
        # convert to pandas dataframe
        pred_as_df = pred_xa_col.pd_dataframe().swapaxes(1, 0).reset_index()
        pred_as_df.rename(
            columns={pred_as_df.columns[-1]: col_name}, inplace=True
        )
        # merge
        self.station_groups = (
            self.station_groups.reset_index()
            .merge(
                pred_as_df, how="left", left_on="group", right_on="component"
            )
            .drop("component", axis=1)
        ).set_index("group")

    def create_base_station(self, gt_col="gt"):
        assert gt_col in self.station_groups, f"must add column {gt_col} first"
        # get only the necessary columns and rows for the gt stations
        # excluding groups
        self.base_station = self.station_groups.loc[
            ~self.station_groups.index.str.contains("Group"),
            [gt_col, "x", "y"],
        ]
        self.base_station["dist"] = (
            self.base_station[gt_col] / self.base_station[gt_col].sum()
        )
        return self.base_station

    def save(self, save_path):
        # self.station_groups.to_csv(
        #     os.path.join(save_path, "station_groups.csv")
        # )
        with open(save_path, "w") as outfile:
            json.dump(self.hier, outfile)
