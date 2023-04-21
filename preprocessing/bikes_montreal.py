import os
import numpy as np
import pandas as pd
import geopandas as gpd


def read_all_demand(base_path="data/bikes_montreal"):
    demand_df = []
    for folder in ["2014", "2015", "2016", "2017"]:
        stations = (
            pd.read_csv(f"{base_path}/{folder}/Stations_{folder}.csv")
            .drop("name", axis=1)
            .set_index("code")
        )
        for f in os.listdir(os.path.join(base_path, folder)):
            if not f.startswith("OD_"):
                continue
            d = pd.read_csv(os.path.join(base_path, folder, f))
            d = d.merge(
                stations,
                left_on="start_station_code",
                right_index=True,
                how="left",
            )
            d = d.drop(
                [
                    "end_date",
                    "is_member",
                    "end_station_code",
                    "start_station_code",
                ],
                axis=1,
            ).rename(
                columns={
                    "start_date": "start_time",
                    "latitude": "start_x",
                    "longitude": "start_y",
                }
            )
            demand_df.append(d)
    demand_df = pd.concat(demand_df)
    print(f"Read {len(demand_df)} records")
    return demand_df


def project_locations(demand_df, target_crs="EPSG:2138"):
    demand_df = gpd.GeoDataFrame(
        demand_df,
        geometry=gpd.points_from_xy(demand_df["start_y"], demand_df["start_x"]),
    )
    demand_df.crs = "EPSG:4326"
    demand_df.to_crs(target_crs, inplace=True)
    demand_df["start_x"] = demand_df.geometry.x
    demand_df["start_y"] = demand_df.geometry.y
    return demand_df


def derive_stations(demand_df):
    nr_stations = demand_df.groupby(["start_x", "start_y"])[
        "duration_sec"
    ].count()
    station_locations = (
        pd.DataFrame(nr_stations).reset_index().drop(["duration_sec"], axis=1)
    )
    station_locations["station_id"] = np.arange(len(station_locations))
    # merge with station IDs
    demand_df = demand_df.merge(
        station_locations,
        left_on=["start_x", "start_y"],
        right_on=["start_x", "start_y"],
        how="left",
    )
    demand_df.drop(["start_x", "start_y", "geometry"], axis=1, inplace=True)
    demand_df.sort_values("start_time", inplace=True)
    return demand_df, station_locations


def make_test_excerpt(demand_df, stations):
    demand_df.sort_values("start_time", inplace=True)
    excerpt = demand_df[:500000]
    # reduce to occuring stations
    occurring_ids = excerpt["station_id"].unique()
    stations_pruned = stations[stations["station_id"].isin(occurring_ids)]
    column_mapping = {col: i for i, col in enumerate(occurring_ids)}
    # map the station IDs
    stations_pruned["station_id"] = stations_pruned["station_id"].map(
        column_mapping
    )
    excerpt["station_id"] = excerpt["station_id"].map(column_mapping)
    return excerpt, stations_pruned


if __name__ == "__main__":
    demand_df = read_all_demand()
    demand_df = project_locations(demand_df)
    demand_df, stations = derive_stations(demand_df)

    stations.to_csv("data/bikes_montreal/stations.csv", index=False)
    demand_df.to_csv("data/bikes_montreal/data.csv", index=False)

    excerpt, stations_pruned = make_test_excerpt(demand_df, stations)
    excerpt.to_csv("data/bikes_montreal/test_data.csv", index=False)
    stations_pruned.to_csv("data/bikes_montreal/test_stations.csv", index=False)
