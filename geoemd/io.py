import pandas as pd


def load_stations(in_path_stations: str) -> pd.DataFrame:
    stations_locations = pd.read_csv(in_path_stations).set_index("station_id")
    return stations_locations


def load_time_series_data(in_path_data: str) -> pd.DataFrame:
    """Load time series and, if in sparse representation, pivot"""
    demand_df = pd.read_csv(in_path_data)
    demand_df["timeslot"] = pd.to_datetime(demand_df["timeslot"])
    # OPTIONAL: make even smaller excerpt
    # stations_included = stations_locations.sample(50).index
    # stations_locations = stations_locations[
    #     stations_locations.index.isin(stations_included)
    # ]
    # # reduce demand matrix shape
    # demand_agg = demand_agg[stations_included]
    # print(demand_agg.shape)
    # pivot if necessary
    if "station_id" in demand_df.columns:
        print("pivoting")
        demand_df = demand_df.pivot(
            index="timeslot", columns="station_id", values="count"
        ).fillna(0)
        demand_df = (
            demand_df.reset_index()
            .rename_axis(None, axis=1)
            .set_index("timeslot")
        )
    else:
        demand_df.set_index("timeslot", inplace=True)
    print("Demand matrix initially", demand_df.shape)
    return demand_df
