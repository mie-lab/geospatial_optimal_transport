import os
import pandas as pd
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate
from darts.models import LinearRegressionModel, XGBModel, NHiTSModel, Croston
from darts.dataprocessing.transformers import MinTReconciliator

from hierarchy_utils import add_demand_groups
from station_hierarchy import StationHierarchy
from utils import get_error_group_level
from visualization import plot_error_evolvement
import warnings

warnings.filterwarnings("ignore")

TRAIN_CUTOFF = 0.9
TEST_SAMPLES = 50  # number of time points where we start a prediction
STEPS_AHEAD = 3

model_class_dict = {
    "linear": (LinearRegressionModel, {"lags": 5}),
    "xgb": (XGBModel, {"lags": 5}),
    "nhits": (
        NHiTSModel,
        {"input_chunk_length": 5, "output_chunk_length": 3, "n_epochs": 3},
    ),
    "croston": (Croston, {}),
}

np.random.seed(42)


def clean_single_pred(pred, pred_or_gt="pred"):
    result_as_df = pred.pd_dataframe().swapaxes(1, 0).reset_index()
    result_as_df.rename(
        columns={c: i for i, c in enumerate(result_as_df.columns[1:])},
        inplace=True,
    )
    result_as_df = pd.melt(result_as_df, id_vars=["component"]).rename(
        {"component": "group", "value": pred_or_gt, "timeslot": "steps_ahead"},
        axis=1,
    )
    return result_as_df


def load_data(in_path_data, in_path_stations):
    demand_df = pd.read_csv(in_path_data)
    stations_locations = pd.read_csv(in_path_stations).set_index("station_id")
    demand_df["timeslot"] = pd.to_datetime(demand_df["timeslot"])

    # create matrix
    demand_agg = demand_df.pivot(
        index="timeslot", columns="station_id", values="count"
    ).fillna(0)
    print("Demand matrix", demand_agg.shape)
    # OPTIONAL: make even smaller excerpt
    # stations_included = stations_locations.sample(50).index
    # stations_locations = stations_locations[
    #     stations_locations.index.isin(stations_included)
    # ]
    # # reduce demand matrix shape
    # demand_agg = demand_agg[stations_included]
    # print(demand_agg.shape)
    return demand_agg, stations_locations


def test_models(
    demand_agg, stations_locations, out_path, models_to_test=["linear_multi_no"]
):
    station_hierarchy = StationHierarchy()
    station_hierarchy.init_from_station_locations(stations_locations)
    demand_agg = add_demand_groups(demand_agg, station_hierarchy.hier)

    # initialize time series with hierarchy
    shared_demand_series = TimeSeries.from_dataframe(demand_agg, freq="1h")
    shared_demand_series = shared_demand_series.with_hierarchy(
        station_hierarchy.get_darts_hier()
    )
    # split train and val
    train_cutoff = int(TRAIN_CUTOFF * len(demand_agg))
    train = shared_demand_series[:train_cutoff]

    # select TEST_SAMPLES random time points during val time
    assert TEST_SAMPLES < len(shared_demand_series) - train_cutoff - STEPS_AHEAD
    random_val_samples = np.random.choice(
        np.arange(train_cutoff, len(shared_demand_series) - STEPS_AHEAD),
        TEST_SAMPLES,
        replace=False,
    )

    # Add gt
    gt_res_dfs = []
    for val_sample in random_val_samples:
        gt_steps_ahead = shared_demand_series[
            val_sample : val_sample + STEPS_AHEAD
        ]
        gt_as_df = clean_single_pred(gt_steps_ahead, pred_or_gt="gt")
        gt_as_df["val_sample_ind"] = val_sample - train_cutoff
        gt_res_dfs.append(gt_as_df)
    gt_res_dfs = pd.concat(gt_res_dfs).reset_index(drop=True)
    gt_res_dfs.to_csv(os.path.join(out_path, "gt.csv"), index=False)

    # Get predictions for each model and save them
    for model_name in models_to_test:
        tic = time.time()

        # get parameters
        model_class_name, multi_vs_ind, do_reconcile = model_name.split("_")
        ModelClass, params = model_class_dict[model_class_name]
        print("Train and test:", model_class_name, multi_vs_ind, do_reconcile)

        # fit model
        if multi_vs_ind == "multi":
            model = ModelClass(**params)
            model.fit(train)
        else:  # independent forecast
            fitted_models = []
            for component in shared_demand_series.components:
                model = ModelClass(**params)
                model.fit(train[component])
                fitted_models.append(model)

        model_res_dfs = []
        for val_sample in random_val_samples:
            if multi_vs_ind == "multi":
                pred_raw = model.predict(
                    n=STEPS_AHEAD, series=shared_demand_series[:val_sample]
                )
            else:
                # if the models were fitted independently, collect the results
                preds_collect = []
                for fitted_model, component in zip(
                    fitted_models, shared_demand_series.components
                ):
                    preds_collect.append(
                        fitted_model.predict(
                            n=STEPS_AHEAD,
                            series=shared_demand_series[component][:val_sample],
                        )
                    )
                pred_raw = concatenate(preds_collect, axis="component")

            # potentially reconcile them
            if do_reconcile == "reconcile":
                reconciliator = MinTReconciliator(method="wls_val")
                reconciliator.fit(train)
                pred = reconciliator.transform(pred_raw)
            else:
                pred = pred_raw

            # transform to flat df
            result_as_df = clean_single_pred(pred)
            # add info about val sample
            result_as_df["val_sample_ind"] = val_sample - train_cutoff
            model_res_dfs.append(result_as_df)

        model_res_dfs = pd.concat(model_res_dfs)
        model_res_dfs.to_csv(
            os.path.join(out_path, f"{model_name}.csv"), index=False
        )
        print("Finished, runtime:", round(time.time() - tic, 2))
    # save the station hierarchy
    station_hierarchy.save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default="../data/bikes_montreal/test_pickup.csv",
    )
    parser.add_argument(
        "-s",
        "--station_path",
        type=str,
        default="../data/bikes_montreal/test_stations.csv",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="outputs/test",
    )
    args = parser.parse_args()
    in_path_data = args.data_path
    in_path_stations = args.station_path
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    demand_agg, stations_locations = load_data(in_path_data, in_path_stations)
    test_models(
        demand_agg,
        stations_locations,
        out_path,
        models_to_test=["nhits_multi_no", "linear_multi_no"],
    )
