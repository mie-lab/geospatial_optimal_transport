import os
import pandas as pd
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import MinTReconciliator
from scipy.spatial.distance import cdist

from model_wrapper import ModelWrapper, CovariateWrapper
from hierarchy_utils import add_demand_groups
from station_hierarchy import StationHierarchy
from utils import argument_parsing, construct_name
from sinkhorn_loss import SinkhornLoss, DistributionMSE
from config import (
    STEPS_AHEAD,
    TRAIN_CUTOFF,
    TEST_SAMPLES,
    MAX_RENTALS,
    model_class_dict,
)
import warnings

warnings.filterwarnings("ignore")

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
    result_as_df[pred_or_gt].clip(0, MAX_RENTALS, inplace=True)
    return result_as_df


def load_data(in_path_data, in_path_stations, pivot=False):
    demand_df = pd.read_csv(in_path_data)
    stations_locations = pd.read_csv(in_path_stations).set_index("station_id")
    demand_df["timeslot"] = pd.to_datetime(demand_df["timeslot"])
    # OPTIONAL: make even smaller excerpt
    # stations_included = stations_locations.sample(50).index
    # stations_locations = stations_locations[
    #     stations_locations.index.isin(stations_included)
    # ]
    # # reduce demand matrix shape
    # demand_agg = demand_agg[stations_included]
    # print(demand_agg.shape)
    return demand_df, stations_locations


def test_models(
    shared_demand_series,
    out_path,
    multi_vs_ind="multi",
    model="linear",
    reconcile=0,
    model_out_name="test_model",
    **kwargs,
):
    # split train and val
    train_cutoff = int(TRAIN_CUTOFF * len(shared_demand_series))
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

    # get past covariates
    cov_lag = (
        kwargs["lags_past_covariates"] if model != "nhits" else kwargs["lags"]
    )
    covariate_wrapper = CovariateWrapper(
        shared_demand_series,
        train_cutoff,
        lags_past_covariates=cov_lag,
        dt_covariates=True,
    )

    # Get predictions for each model and save them
    # for model_name in [model_class]:
    tic = time.time()

    # fit model
    if multi_vs_ind == "multi":
        regr = ModelWrapper(model, covariate_wrapper, **kwargs)
        regr.fit(train)
    else:  # independent forecast
        fitted_models = []
        for component in shared_demand_series.components:
            regr = ModelWrapper(model, covariate_wrapper, **kwargs)
            regr.fit(train[component])
            fitted_models.append(regr)

    # predict
    model_res_dfs = []
    for val_sample in random_val_samples:
        if multi_vs_ind == "multi":
            pred_raw = regr.predict(
                n=STEPS_AHEAD,
                series=shared_demand_series[:val_sample],
                val_index=val_sample,
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
                        val_index=val_sample,
                    )
                )
            pred_raw = concatenate(preds_collect, axis="component")

        # potentially reconcile them
        if reconcile:
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
        os.path.join(out_path, f"{model_out_name}.csv"), index=False
    )
    print("Finished, runtime:", round(time.time() - tic, 2))


if __name__ == "__main__":
    args = argument_parsing()
    in_path_data = args.data_path
    in_path_stations = args.station_path
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    # TODO: set pivot argument of load_data
    demand_agg, stations_locations = load_data(in_path_data, in_path_stations)

    # construct hierarchy
    if args.hierarchy:
        station_hierarchy = StationHierarchy()
        if "0" in demand_agg.columns:
            demand_agg.drop("0", axis=1, inplace=True)
            stations_locations = stations_locations[
                stations_locations.index != 0
            ]
        station_hierarchy.init_from_station_locations(stations_locations)
        demand_agg = add_demand_groups(demand_agg, station_hierarchy.hier)

        # initialize time series with hierarchy
        shared_demand_series = TimeSeries.from_dataframe(
            demand_agg,
            freq="1h",
            hierarchy=station_hierarchy.get_darts_hier(),
            fillna_value=0,
        )
    else:
        # pivot if necessary
        if "station_id" in demand_agg.columns:
            print("pivoting")
            demand_agg = demand_agg.pivot(
                index="timeslot", columns="station_id", values="count"
            ).fillna(0)
            demand_agg = (
                demand_agg.reset_index()
                .rename_axis(None, axis=1)
                .set_index("timeslot")
            )
        else:
            demand_agg.set_index("timeslot", inplace=True)
        print("Demand matrix", demand_agg.shape)

        shared_demand_series = TimeSeries.from_dataframe(
            demand_agg, freq="1h", fillna_value=0
        )

    training_kwargs = vars(args)

    # Initialize loss function
    if args.x_loss_function == "sinkhorn":
        # sort stations by the same order as the demand columns
        station_coords = stations_locations.loc[
            demand_agg.columns, ["x", "y"]
        ].values
        station_cdist = cdist(station_coords, station_coords)
        station_cdist = station_cdist / np.max(station_cdist)
        training_kwargs["loss_fn"] = SinkhornLoss(station_cdist)
    elif args.x_loss_function == "distribution":
        training_kwargs["loss_fn"] = DistributionMSE()

    # Run model comparison
    test_models(
        shared_demand_series,
        out_path,
        model_out_name=construct_name(args),
        **training_kwargs,
    )

    if args.hierarchy:
        # save the station hierarchy
        station_hierarchy.save(out_path)
