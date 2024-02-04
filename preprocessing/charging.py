import pandas as pd

# NOTE: The Jupyter notebook by Arthur et al was used to preprocess the data:
# https://github.com/arthur-75/Smarter-Mobility-Data-Challenge/blob/main/notebook/cleaning.ipynb
# THe output of their preprocessing pipeline is loaded here
train_cleaned = pd.read_csv(
    "data/raw_charging_stations/train_preprocessed_arthur_2.csv"
)

train_raw = pd.read_csv("data/raw_charging_stations/train.csv")

train_cleaned["date"] = pd.to_datetime(train_cleaned["date"])
train_raw["date"] = pd.to_datetime(train_raw["date"])


# merge the preprocessed version with the raw version to correct some things
train_merged = train_cleaned.merge(
    train_raw[["Station", "date", "Charging", "Passive"]],
    left_on=["Station", "date"],
    right_on=["Station", "date"],
    how="left",
)

# fill nans in the raw data with the preprocessed data where NaNs are filled
train_merged.loc[
    ~pd.isna(train_merged["Charging"]), "Charging_f_m"
] = train_merged.loc[~pd.isna(train_merged["Charging"]), "Charging"]
train_merged.loc[
    ~pd.isna(train_merged["Passive"]), "Passive_f_m"
] = train_merged.loc[~pd.isna(train_merged["Passive"]), "Passive"]

# Here: "count" corresponds to the occupied charging stations
train_merged["count"] = (
    train_merged["Charging_f_m"] + train_merged["Passive_f_m"]
)
train_merged = train_merged.drop(
    ["Charging", "Unnamed: 0", "Passive", "Charging_f_m", "Passive_f_m"], axis=1
).rename({"Station": "station_id", "date": "timeslot"}, axis=1)
# round count
train_merged["count"] = train_merged["count"].round()
train_merged.dropna(inplace=True)


# get station ID
train_merged["station_id"] = train_merged["station_id"].str.split("*").str[-2]

# remove data where no cars are charging
mytrain = train_merged.pivot(
    index="timeslot", columns="station_id", values="count"
)
charging_per_station = mytrain.sum()
charging_per_station = charging_per_station[charging_per_station > 0]
included_stations = charging_per_station.index

train_merged = train_merged[train_merged["station_id"].isin(included_stations)]
train_merged.to_csv("data/charging/data_check.csv", index=False)

# reduce stations:
stations = pd.read_csv("data/raw_charging_stations/stations.csv")
stations = stations[stations["station_id"].isin(included_stations.astype(int))]
stations.to_csv("data/charging/stations_check.csv", index=False)
