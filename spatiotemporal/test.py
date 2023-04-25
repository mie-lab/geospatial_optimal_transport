import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate
from darts.models import LinearRegressionModel, XGBModel
from darts.dataprocessing.transformers import MinTReconciliator

from hierarchy_utils import aggregate_bookings, add_demand_groups
from station_hierarchy import StationHierarchy
from utils import get_error_group_level
from visualization import plot_error_evolvement

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
station_hierarchy = StationHierarchy()
station_hierarchy.init_from_station_locations(stations_locations)
demand_agg = aggregate_bookings(demand_df)
demand_agg = add_demand_groups(demand_agg, station_hierarchy.hier)

# train model
tourism_series = TimeSeries.from_dataframe(demand_agg)
tourism_series = tourism_series.with_hierarchy(
    station_hierarchy.get_darts_hier()
)
train, val = tourism_series[:-8], tourism_series[-8:]

# Model comparison
# For now, only compare linear and xgb in different configurations
model_class_dict = {"linear": LinearRegressionModel, "xgb": XGBModel}
params = {"lags": 5}

comparison = pd.DataFrame()
best_mean_error = np.inf
# for ModelClass, model_name, params in zip(
for model_name in [
    "linear_multi_no",
    "linear_ind_no",
    "linear_ind_reconcile",
    "xgb_multi_no",
    "xgb_multi_reconcile",
    "xgb_ind_no",
    "xgb_ind_reconcile",
]:
    # get parameters
    model_class_name, multi_vs_ind, do_reconcile = model_name.split("_")
    ModelClass = model_class_dict[model_class_name]
    print(model_class_name, multi_vs_ind, do_reconcile)

    if multi_vs_ind == "multi":
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

    if do_reconcile == "reconcile":
        reconciliator = MinTReconciliator(method="wls_val")
        reconciliator.fit(train)
        pred = reconciliator.transform(pred_raw)
    else:
        pred = pred_raw

    for steps_ahead in range(len(val)):
        station_hierarchy.add_pred(
            pred[steps_ahead], f"pred_{model_name}_{steps_ahead}"
        )

    # check errors
    error_evolvement = get_error_group_level(
        pred, val, station_hierarchy.station_groups
    )
    # add to comparison
    comparison[model_name] = error_evolvement[:, 1]
    plot_error_evolvement(
        error_evolvement, os.path.join(out_path, f"errors_{model_name}.png")
    )
    current_mean_error = np.mean(error_evolvement[:, 1])
    print(model_class_name, "- mean error:", current_mean_error)
    if current_mean_error < best_mean_error:
        best_mean_error = current_mean_error
        best_model = model_name

comparison.index = error_evolvement[:, 0]

comparison.to_csv(os.path.join(out_path, "model_comparison.csv"))
plt.figure(figsize=(6, 6))
comparison.plot()
plt.savefig(os.path.join(out_path, "comparison.png"))

# Do optimal transport stuff with best pred
for steps_ahead in range(len(val)):
    station_hierarchy.add_pred(val[steps_ahead], f"gt_{steps_ahead}")

station_hierarchy.save(os.path.join("outputs", "test1"))
