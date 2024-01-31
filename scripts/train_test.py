import os
import pandas as pd
import time
import numpy as np
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import MinTReconciliator
from scipy.spatial.distance import cdist

from geoemd.model_wrapper import ModelWrapper, CovariateWrapper
from geoemd.hierarchy.hierarchy_utils import construct_series_with_hierarchy
from geoemd.hierarchy.clustering_hierarchy import SpatialClustering
from geoemd.utils import (
    argument_parsing,
    construct_name,
    get_dataset_name,
    get_emd_loss_function,
    space_cost_matrix,
)
from geoemd.io import load_time_series_data, load_stations
from geoemd.parameter_optimization import OptunaOptimizer
from geoemd.loss.distribution_loss import StepwiseCrossentropy, DistributionMSE
from geoemd.config import (
    STEPS_AHEAD,
    TRAINTEST_SPLIT,
    TRAINVAL_SPLIT,
    TEST_SAMPLES,
    MAX_COUNT,
    AGG_FUNCTION,
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


def train_and_test(
    shared_demand_series,
    out_path,
    multi_vs_ind="multi",
    model="linear",
    norm_factor=10,
    reconcile=0,
    ordered_test_samples=False,
    optimize_optuna=False,
    **kwargs,
):
    # normalize whole time series
    shared_demand_series = shared_demand_series / norm_factor

    # split train and val
    val_cutoff = int(TRAINVAL_SPLIT * len(shared_demand_series))  # 0.8
    train_cutoff = int(TRAINTEST_SPLIT * len(shared_demand_series))  # 0.9
    train = shared_demand_series[:val_cutoff]
    val = shared_demand_series[val_cutoff:train_cutoff]

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

    if optimize_optuna:
        param_optim = OptunaOptimizer(**kwargs)
        param_optim(train, val)
        return 0

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
        val_cutoff,
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
        regr.fit(train, val_series=val)
    else:  # independent forecast
        fitted_models = []
        for component in shared_demand_series.components:
            regr = ModelWrapper(model, covariate_wrapper, **kwargs)
            regr.fit(train[component])
            fitted_models.append(regr)

    if DEBUG:
        return 0

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

    demand_agg = load_time_series_data(in_path_data)

    # if we want to train with a darts hierarchy
    if args.hierarchy:
        assert (
            args.y_clustermethod == "agg" and dataset != "traffic"
        ), "Only agglomerative clustering implemented and not for traffic data"
        main_time_series = construct_series_with_hierarchy(
            demand_agg, in_path_stations, FREQUENCY[dataset]
        )
    else:
        # for just clustering and training on the clustered data
        if args.y_clustermethod is not None:
            station_hierarchy = SpatialClustering(
                in_path_stations, is_cost_matrix=(dataset == "traffic")
            )
            station_hierarchy(
                clustering_method=args.y_clustermethod,
                n_clusters=args.y_cluster_k,
            )
            # transform the demand to get the grouped df
            demand_agg = station_hierarchy.transform_demand(
                demand_agg,
                hierarchy=args.hierarchy,
                agg_func=AGG_FUNCTION[dataset],
            )
        # no clustering
        print("time series after preprocessing", demand_agg.shape)
        main_time_series = TimeSeries.from_dataframe(
            demand_agg, freq=FREQUENCY[dataset], fillna_value=0
        )

    # get norm factor
    norm_factor = np.quantile(demand_agg.values, 0.95)
    print("Normalize dividing by", norm_factor)
    # derive name for saving
    out_name, training_kwargs = construct_name(args)

    # Initialize loss function
    if "emd" in args.x_loss_function:
        if args.y_clustermethod is not None:
            # get dist matrix (between groups!)
            time_dist_matrix = station_hierarchy.get_clustered_cost_matrix(
                demand_agg.columns
            )
        # if we don't have a clustering, we need to create the cost matrix
        else:
            # load stations -> either cost matrix or list of spatial coords
            stations = load_stations(in_path_stations)
            if dataset == "traffic":
                # simply the values of the cost matrix
                assert all(stations.index.astype(str) == demand_agg.columns)
                time_dist_matrix = stations.values
            else:
                station_coords = stations.loc[
                    demand_agg.columns, ["x", "y"]
                ].values
                # pairwise distance between stations (in terms of travel time)
                time_dist_matrix = space_cost_matrix(station_coords)
        # sort stations by the same order as the demand columns
        # TODO: , speed_factor=SPEED_FACTOR[dataset], quadratic=False
        training_kwargs["loss_fn"] = get_emd_loss_function(
            args.x_loss_function, time_dist_matrix
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
        main_time_series,
        norm_factor=norm_factor,
        ordered_test_samples=ordered_test_samples,
        **training_kwargs,
    )

    if args.y_clustermethod is not None:
        # save the station hierarchy
        station_hierarchy.save(
            os.path.join(out_path, out_name + "_hierarchy.json")
        )
