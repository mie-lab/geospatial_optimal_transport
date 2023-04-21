import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate
from darts.datasets import AustralianTourismDataset
import pandas as pd
from darts.models import LinearRegressionModel, XGBModel
from darts.metrics import mae
from darts.dataprocessing.transformers import MinTReconciliator

from hierarchy import (
    stations_to_hierarchy,
    aggregate_bookings,
    hier_to_darts,
    add_demand_groups,
)
from utils import (
    get_error_group_level,
    get_children_hierarchy,
    create_groups_with_pred,
)
from visualization import plot_error_evolvement
from optimal_transport import transport_equal_dist, transport_from_centers

in_path_data = "../data/bikes_montreal/test_data.csv"
in_path_stations = "../data/bikes_montreal/test_stations.csv"
out_path = "outputs"
os.makedirs(out_path, exist_ok=True)

demand_df = pd.read_csv(in_path_data)
stations_locations = pd.read_csv(in_path_stations).set_index("station_id")
demand_df["start_time"] = pd.to_datetime(demand_df["start_time"])

# make even smaller excerpt
max_station = 50
demand_df = demand_df[demand_df["station_id"] < max_station]
stations_locations = stations_locations[stations_locations.index < max_station]

# run the preprocessing
station_groups, hier = stations_to_hierarchy(stations_locations)
demand_agg = aggregate_bookings(demand_df)
demand_agg = add_demand_groups(demand_agg, hier)
darts_hier = hier_to_darts(hier)

# train model
tourism_series = TimeSeries.from_dataframe(demand_agg)
tourism_series = tourism_series.with_hierarchy(darts_hier)
train, val = tourism_series[:-8], tourism_series[-8:]

# Model comparison
comparison = pd.DataFrame()
best_mean_error = np.inf
for ModelClass, model_name, params in zip(
    [LinearRegressionModel],  # XGBModel
    ["linear_multi"],  # , "linear_reconcile"
    [{"lags": 5}],  # , {"lags": 5}
):
    if "multi" in model_name:
        model = ModelClass(**params)
        model.fit(train)
        pred_raw = model.predict(n=len(val))
    else:  # independent forecast
        preds_collect = []
        for component in tourism_series.components:
            model = ModelClass(**params)
            model.fit(train[component])
            preds_collect.append(model.predict(n=len(val)))
        pred_raw = concatenate(preds_collect, axis="component")

    if "reconcile" in model_name:
        reconciliator = MinTReconciliator(method="wls_val")
        reconciliator.fit(train)
        pred = reconciliator.transform(pred_raw)
    else:
        pred = pred_raw

    # check errors
    error_evolvement = get_error_group_level(pred, val, station_groups)
    # add to comparison
    comparison[model_name] = error_evolvement[:, 1]
    plot_error_evolvement(
        error_evolvement, os.path.join(out_path, f"errors_{model_name}.png")
    )
    current_mean_error = np.mean(error_evolvement[:, 1])
    if current_mean_error < best_mean_error:
        best_mean_error = current_mean_error
        best_pred = pred.copy()

comparison.index = error_evolvement[:, 0]

comparison.to_csv(os.path.join(out_path, "model_comparison.csv"))
plt.figure(figsize=(6, 6))
comparison.plot()
plt.savefig(os.path.join(out_path, "comparison.png"))

# continue with best pred
pred = best_pred
step_ahead = 0

# TODO: maybe disentangle, make prediction column per step ahead, etc
groups_with_pred = create_groups_with_pred(
    pred, val, step_ahead, station_groups
)

# generate groups with pred TODO: put in function -> group_with_pred not needed
base_station = groups_with_pred.loc[
    ~groups_with_pred.index.str.contains("Group"), ["gt", "start_x", "start_y"]
]
base_station_coords = base_station[["start_x", "start_y"]].values.astype(float)
base_station["dist"] = base_station["gt"] / base_station["gt"].sum()
base_station_dist = base_station["dist"].values

transport_from_centers(groups_with_pred, base_station, hier)
transport_equal_dist(groups_with_pred, base_station, hier)
