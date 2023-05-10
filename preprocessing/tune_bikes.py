import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
import os


def find_representative_stations(stations, cutoff=300):
    print("Mean and median of raw stations")
    target_mean = stations["demand"].mean()
    target_median = stations["demand"].median()
    print(target_mean, target_median)
    print("-------------------------------")

    xy = stations[["x", "y"]].values
    distances = cdist(xy, xy)
    nns = np.argsort(distances, axis=1)
    rand_samples = np.random.permutation(len(nns))
    for rand_sample in rand_samples[:cutoff]:
        rand_neighbors = nns[rand_sample, :5]
        station_subset = stations.iloc[rand_neighbors]
        subset_mean, subset_median = (
            station_subset["demand"].mean(),
            station_subset["demand"].median(),
        )
        if (abs(subset_mean - target_mean) < 0.1 * target_mean) and (
            abs(subset_median - target_median) < 0.1 * target_mean
        ):
            print(rand_neighbors.tolist())
            print(subset_mean, subset_median)
            print()
    exit()


if __name__ == "__main__":
    base_path = "data/bikes_montreal/"
    # Read full demand and stations
    in_path_data = os.path.join(base_path, "pickup.csv")
    demand_all = pd.read_csv(in_path_data).sort_values("timeslot")
    demand_all["timeslot"] = pd.to_datetime(demand_all["timeslot"])
    in_path_stations = os.path.join(base_path, "stations.csv")
    stations = pd.read_csv(in_path_stations, index_col="station_id")

    # restrict to one year - otherwise I might have weird gap effects TODO
    demand_all = demand_all[
        demand_all["timeslot"] < pd.to_datetime("2015-01-01")
    ]

    # pivot demand to gt demand per station
    demand_pivoted = demand_all.pivot(
        index="timeslot", columns="station_id", values="count"
    ).fillna(0)
    demand_per_station = demand_pivoted.sum()
    stations["demand"] = demand_per_station
    # drop the ones without demand during that time
    stations.dropna(inplace=True)

    # potentially find representative stations
    # IF not found yet:
    # find_representative_stations(stations)
    # IF already selected the most representative stations for tuning
    BEST_TUNE_STATIONS = [185, 425, 156, 179, 443]
    station_subset = stations.iloc[BEST_TUNE_STATIONS]
    print(
        "Mean and Median",
        station_subset["demand"].mean(),
        station_subset["demand"].median(),
    )

    # Add sum of these stations as station 0
    station_subset.loc[0] = {
        "x": station_subset["x"].mean(),
        "y": station_subset["y"].mean(),
        "demand": station_subset["demand"].sum(),
    }
    # Save stations
    station_subset.to_csv(os.path.join(base_path, "tune_stations.csv"))

    # Restrict demand to the one at these stations
    best_tune_stations_ids = station_subset.index
    demand_tune_stations = demand_all[
        demand_all["station_id"].isin(best_tune_stations_ids)
    ]

    demand_tune_stations = demand_tune_stations.pivot(
        index="timeslot", columns="station_id", values="count"
    ).fillna(0)
    # add the sum of these stations as column "0"
    demand_tune_stations[0] = demand_tune_stations.sum(axis=1)
    demand_tune_stations.reset_index().to_csv(
        os.path.join(base_path, "tune_pickup.csv"), index=False
    )
