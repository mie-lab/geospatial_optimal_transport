import os
import pandas as pd
import time
import numpy as np
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import MinTReconciliator
from scipy.spatial.distance import cdist

from geoemd.model_wrapper import ModelWrapper, CovariateWrapper
from geoemd.hierarchy.hierarchy_utils import add_demand_groups
from geoemd.hierarchy.full_station_hierarchy import FullStationHierarchy
from geoemd.hierarchy.clustering_hierarchy import SpatialClustering
from geoemd.utils import (
    argument_parsing,
    construct_name,
    get_dataset_name,
    spacetime_cost_matrix,
)
from geoemd.loss.sinkhorn_loss import SinkhornUnbalanced, CombinedLoss
from geoemd.loss.distribution_loss import StepwiseCrossentropy, DistributionMSE
from geoemd.loss.unbalanced import UnbalancedSinkhorn
from geoemd.config import (
    STEPS_AHEAD,
    TRAINTEST_SPLIT,
    TEST_SAMPLES,
    MAX_COUNT,
    SPEED_FACTOR,
    FREQUENCY,
)
import warnings

DEBUG = False
warnings.filterwarnings("ignore")

np.random.seed(42)


def clean_single_pred(pred, pred_or_gt="pred", clip=True, apply_exp=False):
    result_as_df = pred.pd_dataframe().swapaxes(1, 0).reset_index()
    result_as_df.rename(
        columns={c: i for i, c in enumerate(result_as_df.columns[1:])},
        inplace=True,
    )
    result_as_df = pd.melt(result_as_df, id_vars=["component"]).rename(
        {"component": "group", "value": pred_or_gt, "timeslot": "steps_ahead"},
        axis=1,
    )
    if apply_exp:
        result_as_df[pred_or_gt] = np.exp(result_as_df[pred_or_gt])
    if clip:
        result_as_df[pred_or_gt].clip(0, MAX_COUNT, inplace=True)
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
    print("Demand matrix", demand_df.shape)
    return demand_df, stations_locations


def train_and_test(
    shared_demand_series,
    out_path,
    multi_vs_ind="multi",
    model="linear",
    norm_factor=10,
    reconcile=0,
    ordered_test_samples=False,
    **kwargs,
):
    # normalize whole time series
    shared_demand_series = shared_demand_series / norm_factor

    # split train and val
    train_cutoff = int(TRAINTEST_SPLIT * len(shared_demand_series))
    train = shared_demand_series[:train_cutoff]

    # select TEST_SAMPLES random time points during val time
    assert TEST_SAMPLES < len(shared_demand_series) - train_cutoff - STEPS_AHEAD
    # ensure that the test samples are always the same
    np.random.seed(48)
    # either use sorted val samples, or use independent ones
    if ordered_test_samples:
        # at every step, we predict x steps ahead
        random_val_samples = np.arange(
            train_cutoff, train_cutoff + TEST_SAMPLES * 5  # , STEPS_AHEAD
        )
    else:
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

    if DEBUG:
        exit()

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

        # Clean: (transform to df, clip, etc)
        # if loss function is just distribution, we apply exp to the results
        apply_exp = kwargs["x_loss_function"] in ["sinkhorn", "distribution"]
        result_as_df = clean_single_pred(pred, clip=True, apply_exp=apply_exp)
        # add info about val sample
        result_as_df["val_sample_ind"] = val_sample - train_cutoff
        model_res_dfs.append(result_as_df)

    model_res_dfs = pd.concat(model_res_dfs).reset_index(drop=True)
    # re-nomalize and add gt
    model_res_dfs["pred"] *= norm_factor
    assert all(
        model_res_dfs.drop("pred", axis=1) == gt_res_dfs.drop("gt", axis=1)
    )
    model_res_dfs["gt"] = gt_res_dfs["gt"].values * norm_factor
    # save with save name
    model_name = kwargs.get("model_name", "test_model")
    model_res_dfs.to_csv(
        os.path.join(out_path, f"{model_name}.csv"), index=False
    )
    print("Finished, runtime:", round(time.time() - tic, 2))

    regr.save()
    print("Model saved")


if __name__ == "__main__":
    args = argument_parsing()
    in_path_data = args.data_path
    in_path_stations = args.station_path
    out_path = args.out_path
    ordered_test_samples = args.ordered_samples
    os.makedirs(out_path, exist_ok=True)

    dataset = get_dataset_name(in_path_data)

    # TODO: set pivot argument of load_data
    demand_agg, stations_locations = load_data(in_path_data, in_path_stations)

    # construct hierarchy
    if args.hierarchy and args.y_clustermethod == "agg":
        station_hierarchy = FullStationHierarchy()
        if "0" in demand_agg.columns:
            demand_agg.drop("0", axis=1, inplace=True)
            stations_locations = stations_locations[
                stations_locations.index != 0
            ]
        station_hierarchy.init_from_station_locations(stations_locations)
        demand_agg = add_demand_groups(demand_agg, station_hierarchy.hier)
    elif args.y_clustermethod is not None:
        station_hierarchy = SpatialClustering(stations_locations)
        station_hierarchy(
            clustering_method=args.y_clustermethod, n_clusters=args.y_cluster_k
        )
        # transform the demand to get the grouped df
        demand_agg = station_hierarchy.transform_demand(
            demand_agg, hierarchy=args.hierarchy
        )

    if dataset in ["bikes", "bikes_2015", "carsharing"]:
        norm_factor = np.quantile(demand_agg.values, 0.95)
    elif dataset == "charging":
        norm_factor = 3  # at most 3 cars are charging
    else:
        raise NotImplementedError("only bikes and charging implemented")

    # init time series
    if args.hierarchy:
        # initialize time series with hierarchy
        shared_demand_series = TimeSeries.from_dataframe(
            demand_agg,
            freq=FREQUENCY[dataset],
            hierarchy=station_hierarchy.get_darts_hier(),
            fillna_value=0,
        )
    else:
        shared_demand_series = TimeSeries.from_dataframe(
            demand_agg, freq=FREQUENCY[dataset], fillna_value=0
        )

    out_name, training_kwargs = construct_name(args)

    # Initialize loss function
    if "emd" in args.x_loss_function:
        # sort stations by the same order as the demand columns
        if args.y_clustermethod is not None:
            station_coords = station_hierarchy.groups_coordinates.loc[
                demand_agg.columns
            ].values
        else:
            station_coords = stations_locations.loc[
                demand_agg.columns, ["x", "y"]
            ].values
        station_cdist = cdist(station_coords, station_coords)
        station_cdist = station_cdist / np.max(station_cdist)
        if "temporal" in args.x_loss_function:
            spatiotemporal_cost = spacetime_cost_matrix(
                station_cdist,
                time_steps=STEPS_AHEAD,
                speed_factor=SPEED_FACTOR[dataset],
            )
        if args.x_loss_function == "emdbalancedspatial":
            training_kwargs["loss_fn"] = CombinedLoss(station_cdist)
        elif args.x_loss_function == "emdbalancedspatiotemporal":
            # actually combined sinkhorn temporal
            training_kwargs["loss_fn"] = CombinedLoss(
                spatiotemporal_cost, spatiotemporal=True
            )
        elif args.x_loss_function == "emdunbalancedspatial":
            training_kwargs["loss_fn"] = UnbalancedSinkhorn(station_cdist)
            # training_kwargs["pl_trainer_kwargs"] = {"gradient_clip_val": 1}
        elif args.x_loss_function == "emdunbalancedspatiotemporal":
            training_kwargs["loss_fn"] = UnbalancedSinkhorn(
                spatiotemporal_cost, spatiotemporal=True
            )
        else:
            raise NotImplementedError(
                "Must be emdbalancedspatial, emdunbalancedspatial,\
                     emdunbalancedspatiotemporalor emdbalancedspatiotemporal"
            )
    elif args.x_loss_function == "distribution":
        training_kwargs["loss_fn"] = DistributionMSE()
    elif args.x_loss_function == "crossentropy":
        training_kwargs["loss_fn"] = StepwiseCrossentropy()
    elif args.x_loss_function != "basic":
        raise NotImplementedError(
            "Must be basic, an EMD loss function, distribution or crossentropy"
        )

    # Run model comparison
    train_and_test(
        shared_demand_series,
        norm_factor=norm_factor,
        ordered_test_samples=ordered_test_samples,
        **training_kwargs,
    )

    if args.y_clustermethod is not None:
        # save the station hierarchy
        station_hierarchy.save(
            os.path.join(out_path, out_name + "_hierarchy.json")
        )
