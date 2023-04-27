import os
import numpy as np
import pandas as pd
import geopandas as gpd


def deprecated_read_all_demand(base_path="data/bikes_montreal"):
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


def project_locations(station_df, target_crs="EPSG:2138"):
    station_df = gpd.GeoDataFrame(
        station_df,
        geometry=gpd.points_from_xy(
            station_df["longitude"], station_df["latitude"]
        ),
    )
    station_df.crs = "EPSG:4326"
    station_df.to_crs(target_crs, inplace=True)
    station_df["x"] = station_df.geometry.x
    station_df["y"] = station_df.geometry.y
    return station_df.drop(["latitude", "longitude", "geometry"], axis=1)


def deprecated_derive_stations(demand_df):
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


def make_test_excerpt(demand_df, stations, reset_station_ids=False):
    demand_df.sort_values("timeslot", inplace=True)
    excerpt = demand_df[:500000]
    # reduce to occuring stations
    occurring_ids = excerpt["station_id"].unique()
    stations_pruned = stations[stations["station_id"].isin(occurring_ids)]
    if reset_station_ids:
        column_mapping = {col: i for i, col in enumerate(occurring_ids)}
        # map the station IDs
        stations_pruned["station_id"] = stations_pruned["station_id"].map(
            column_mapping
        )
        excerpt["station_id"] = excerpt["station_id"].map(column_mapping)
    return excerpt, stations_pruned


def load_demand(
    base_path="data/bikes_montreal", station_code_mapping={}, agg_by="1h"
):
    pickups, dropoff = [], []
    for folder in ["2014", "2015", "2016", "2017"]:
        print(folder)
        for f in os.listdir(os.path.join(base_path, folder)):
            if not f.startswith("OD_"):
                continue
            demand = pd.read_csv(os.path.join(base_path, folder, f))

            station_mapping_this_year = {
                key[1]: value
                for key, value in station_code_mapping.items()
                if folder < key[0]
            }
            demand["start_station"] = demand["start_station_code"].apply(
                lambda x: station_mapping_this_year.get(x, x)
            )
            demand["end_station"] = demand["end_station_code"].map(
                lambda x: station_mapping_this_year.get(x, x)
            )

            demand["start_slot"] = pd.to_datetime(
                demand["start_date"]
            ).dt.floor(agg_by)
            # I have to sort in order to make the .floor work - wtf
            #         demand.sort_values("end_date", inplace=True, ascending=False, )
            demand["end_slot"] = pd.to_datetime(demand["end_date"])
            # # the following was only a display error
            # if "h" in agg_by:
            #     # aggregating by hour only works with tmp column
            #     demand["tmp"] = demand["end_slot"].dt.hour
            #     demand.sort_values("tmp", inplace=True, ascending=False)
            demand["end_slot"] = demand["end_slot"].dt.floor(agg_by)
            demand.drop(
                [
                    "start_date",
                    "end_date",
                    "duration_sec",
                    "start_station_code",
                    "end_station_code",
                    "is_member",
                    "tmp",
                ],
                axis=1,
                inplace=True,
                errors="ignore",
            )

            pickups.append(
                demand.groupby(["start_slot", "start_station"])
                .agg({"start_station": "count"})
                .rename(columns={"start_station": "count"})
                .reset_index()
                .rename(
                    columns={
                        "start_station": "station_id",
                        "start_slot": "timeslot",
                    }
                )
            )
            dropoff.append(
                demand.groupby(["end_slot", "end_station"])
                .agg({"end_station": "count"})
                .rename(columns={"end_station": "count"})
                .reset_index()
                .rename(
                    columns={
                        "end_station": "station_id",
                        "end_slot": "timeslot",
                    }
                )
            )
    pickups = pd.concat(pickups)
    dropoff = pd.concat(dropoff)
    # for dropoff, it can occur that there are duplicates at the week overlap
    dropoff = (
        dropoff.groupby(["timeslot", "station_id"])
        .agg({"count": "count"})
        .reset_index()
    )
    return pickups, dropoff


def load_stations(
    base_path="data/bikes_montreal",
    cutoff_distance=0.001,
    remove_relocated_stations=False,
):
    """
    Reads stations from different years from the base_path folder and
    merges them. If they are too far apart, the ID is mapped to a new one (as
    if it was a new station)
    Returns: merged stations and dictionary with the ID mapping (before that
    year!)
    """
    station_code_mapping = {}

    for i, folder in enumerate(["2014", "2015", "2016", "2017"]):
        stations = pd.read_csv(
            os.path.join(base_path, folder, f"Stations_{folder}.csv")
        ).drop("name", axis=1)
        if i == 0:
            all_stations = stations
            continue

        # check if the stations diverge
        together = pd.concat([all_stations, stations])
        stds = together.groupby("code").agg(
            {"latitude": "std", "longitude": "std"}
        )
        problematic_stations = stds[
            (
                ~pd.isna(stds["latitude"])
                & (
                    (stds["latitude"] > cutoff_distance)
                    | (stds["longitude"] > cutoff_distance)
                )
            )
        ].index
        print("Problematic stations", len(problematic_stations))
        # map the stations (in all_stations) to new code
        codes_max = all_stations["code"].max()
        new_station_codes = {
            station_no: i + codes_max
            for i, station_no in enumerate(problematic_stations)
        }
        # need to update the all_stations because otherwise we will find the
        # same stations again and again
        station_code_mapping.update(
            {(folder, key): value for key, value in new_station_codes.items()}
        )  # store year to know when the code need to be updated
        problematic_rows_in_table = all_stations["code"].isin(
            problematic_stations
        )
        all_stations.loc[problematic_rows_in_table, "code"] = all_stations.loc[
            problematic_rows_in_table, "code"
        ].map(new_station_codes, na_action="ignore")

        # concat again
        all_stations = pd.concat([all_stations, stations])
        # take the last lat lon for each of them
        all_stations = all_stations.groupby("code").last().reset_index()

    # remove stations that don't apppear in 2017 - doesn't happen
    if remove_relocated_stations:
        print(len(all_stations))
        all_stations = all_stations[all_stations["code"].isin(stations["code"])]
        print(
            "After removing the ones that don't appear in the last year",
            len(all_stations),
        )
    all_stations.rename(columns={"code": "station_id"}, inplace=True)
    return all_stations, station_code_mapping


def check_gaps(pickups, gap="1h"):
    if gap != "1h":
        raise NotImplementedError("only implemented for 1h")
    # make matrix
    pickup_matrix = pickups.pivot(
        index="timeslot", columns="station_id", values="count"
    ).fillna(0)
    # check whether index is sequential (one booking appears in every hour?)
    new_df = pd.DataFrame(
        list(pd.to_datetime(pickup_matrix.index)), columns=["ind"]
    )
    new_df["next"] = new_df["ind"].shift(-1)
    gaps = new_df[
        (new_df["next"] - new_df["ind"]).astype(str) > "0 days 01:00:00"
    ]
    print("number of gaps", len(gaps))
    print(gaps.head())


def fix_pickup_gap(pickup):
    """There is exactly one gap in the pickups. Fill it with a 0"""
    pickup.set_index("timeslot", inplace=True)
    rand_station = pickup.iloc[0]["station_id"]
    pickup.loc[pd.to_datetime("2015-04-21 04:00:00")] = {
        "station_id": rand_station,
        "count": 0,
    }
    return pickup.reset_index()


def to_csv_td(df, out_path, time_col=["timeslot"]):
    for col in time_col:
        df[col] = df[col].astype(str)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    station_df, station_code_mapping = load_stations()
    station_df = project_locations(station_df)
    pickup, dropoff = load_demand(station_code_mapping=station_code_mapping)

    # fix gap
    pickup = fix_pickup_gap(pickup)

    check_gaps(pickup)
    check_gaps(dropoff)

    station_df.to_csv("data/bikes_montreal/stations.csv", index=False)
    to_csv_td(pickup, "data/bikes_montreal/pickup.csv")
    to_csv_td(dropoff, "data/bikes_montreal/dropoff.csv")

    excerpt_pickup, stations_pruned = make_test_excerpt(
        pickup, station_df, reset_station_ids=False
    )
    excerpt_dropoff = dropoff[
        dropoff["timeslot"] < excerpt_pickup["timeslot"].max()
    ]
    to_csv_td(excerpt_pickup, "data/bikes_montreal/test_pickup.csv")
    to_csv_td(excerpt_dropoff, "data/bikes_montreal/test_dropoff.csv")
    stations_pruned.to_csv("data/bikes_montreal/test_stations.csv", index=False)
