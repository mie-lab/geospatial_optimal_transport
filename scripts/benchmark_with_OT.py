# Standard and GIS Modules
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import time

import torch
from esda.moran import Moran
from geoemd.loss.partial_ot import PartialOT
from geoemd.loss.moransi import MoransiLoss
from libpysal.weights import KNN
from scipy.spatial.distance import cdist

# ignore linalg warnings from MGWR package
import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models import *


def get_folds(nr_samples, nr_folds=10):
    fold_inds = np.random.permutation(nr_samples)
    num_per_fold = nr_samples // nr_folds
    train_inds, test_inds = [], []
    for i in range(nr_folds):
        # print("start, end", i*num_per_fold)
        if i < nr_folds - 1:
            test_inds_fold = np.arange(
                i * num_per_fold, (i + 1) * num_per_fold, 1
            )
        else:
            test_inds_fold = np.arange(i * num_per_fold, nr_samples)
        test_inds.append(fold_inds[test_inds_fold])
        train_inds.append(np.delete(fold_inds, test_inds_fold))
    return train_inds, test_inds


def prepare_data(data, target, lon="x", lat="y"):
    """Assumes that all other columns are used as covariates"""
    # covariates = [col for col in data.columns if col not in [lon, lat, target]]
    # return data[covariates], data[target], data[[lon, lat]]
    return data.rename(
        columns={target: "label", lon: "x_coord", lat: "y_coord"}
    )


def add_metrics(test_pred, test_y, coords, res_dict_init, method, runtime):
    print(test_pred.shape, test_y.shape, coords.shape)
    test_pred = test_pred.squeeze()
    test_y = test_y.squeeze()

    res_dict = res_dict_init.copy()
    res_dict["Method"] = method
    res_dict["RMSE"] = mean_squared_error(test_pred, test_y, squared=False)
    res_dict["MAE"] = mean_absolute_error(test_pred, test_y)
    res_dict["R-Squared"] = r2_score(test_y, test_pred)
    res_dict["Runtime"] = runtime

    # Moran's I KNN
    def compute_knn_moransi(
        gt_values, pred_values, weight_matrix=None, stations=None, k=3
    ):
        # with queen
        if weight_matrix is None:
            station_gpd = gpd.GeoDataFrame(
                stations,
                geometry=gpd.points_from_xy(x=stations[:, 0], y=stations[:, 1]),
            )
            weight_matrix = KNN.from_dataframe(station_gpd, k=k)
        # print(gt_values.shape, pred_values.shape, stations.shape)
        return Moran(gt_values - pred_values, weight_matrix).I

    res_dict["moransi_2"] = compute_knn_moransi(
        test_y, test_pred, stations=coords, k=3
    )
    res_dict["moransi_3"] = compute_knn_moransi(
        test_y, test_pred, stations=coords, k=10
    )

    cost_matrix = cdist(coords, coords)

    moransi_obj = MoransiLoss(cost_matrix)

    res_dict["moransi_cost"] = moransi_obj(
        torch.from_numpy(test_pred).unsqueeze(0),
        torch.from_numpy(test_y).unsqueeze(0),
    ).item()

    min_value = min([np.min(test_pred), np.min(test_y)])
    if min_value < 0:
        test_pred = test_pred - min_value
        test_y = test_y - min_value

    test_pred = torch.from_numpy(test_pred).unsqueeze(0)
    test_y = torch.from_numpy(test_y).unsqueeze(0)

    ot_computer = PartialOT(
        cost_matrix, penalty_unb=0, normalize_c=False, compute_exact=True
    )
    res_dict["ot_loss_0"] = ot_computer(test_pred, test_y)

    ot_computer = PartialOT(
        cost_matrix, penalty_unb="max", normalize_c=False, compute_exact=True
    )
    res_dict["ot_loss_max"] = ot_computer(test_pred, test_y)

    ot_computer = PartialOT(
        cost_matrix,
        normalize_c=False,
        penalty_unb=np.quantile(cost_matrix, 0.1),
        compute_exact=True,
    )
    res_dict["ot_loss_low"] = ot_computer(test_pred, test_y)

    return res_dict


def cross_validation(data):
    nr_folds = 5
    train_inds, test_inds = get_folds(len(data), nr_folds=nr_folds)
    res_df = []

    # dataset specific information
    target = dataset_target[DATASET]
    x_coord_name = dataset_x.get(DATASET, "x")
    y_coord_name = dataset_y.get(DATASET, "y")

    # model params --> TODO: grid search
    max_depth = 10
    spatial_neighbors = len(data) // 5  # one fifth of the dataset
    print("Number of neighbors considered for spatial RF:", spatial_neighbors)

    data_renamed = prepare_data(data.copy(), target, x_coord_name, y_coord_name)

    from collections import defaultdict

    pred_per_model = defaultdict(list)
    gt_per_model = defaultdict(list)
    coords_per_model = defaultdict(list)
    for fold in range(nr_folds):
        res_dict_init = {"fold": fold, "max_depth": max_depth}
        train_data = data_renamed.iloc[train_inds[fold]]
        test_data = data_renamed.iloc[test_inds[fold]]
        feat_cols = [
            col
            for col in train_data.columns
            if "coord" not in col and col != "label"
        ]
        coords = test_data[["x_coord", "y_coord"]].values
        # print(
        #     train_x.shape, train_y.shape, train_coords.shape, test_x.shape,
        #     test_y.shape, test_coords.shape
        # )
        for model_function, name in zip(model_function_names, model_names):
            tic = time.time()
            test_pred = model_function(
                train_data.copy(),
                test_data.copy(),
                feat_cols=feat_cols,
            )
            runtime = time.time() - tic
            # # save as pickle
            # import pickle
            # with open(os.path.join(out_path, f"{name}_{fold}.pkl"), "wb") as f:
            #     pickle.dump((test_pred, test_data["label"], coords), f)
            # res_df.append(
            #     add_metrics(
            #         test_pred,
            #         test_data["label"].values,
            #         coords,
            #         res_dict_init,
            #         name,
            #         runtime,
            #     )
            # )
            pred_per_model[name].append(test_pred.squeeze())
            gt_per_model[name].append(test_data["label"].values.squeeze())
            coords_per_model[name].append(coords)
            # print(name, res_df[-1]["R-Squared"])

    res_df = []
    for name in model_names:
        all_pred, all_gt, all_coords = (
            np.concatenate(pred_per_model[name], 0),
            np.concatenate(gt_per_model[name], 0),
            np.concatenate(coords_per_model[name], 0),
        )
        # save as pickle
        import pickle

        with open(
            os.path.join(out_path, f"{DATASET}_cv_{name}.pkl"), "wb"
        ) as f:
            pickle.dump((all_pred, all_gt, all_coords), f)

        res_df.append(
            add_metrics(
                all_pred,
                all_gt,
                all_coords,
                {"fold": 0, "max_depth": 0},
                name,
                0,
            )
        )

    # Finalize results
    res_df = pd.DataFrame(res_df)
    return res_df


out_path = "outputs/spatial_regression"
os.makedirs(out_path, exist_ok=True)

dataset_target = {
    "plants": "richness_species_vascular",
    "meuse": "zinc",
    "atlantic": "Rate",
    "deforestation": "deforestation_quantile",
}

model_function_names = [
    linear_regression,
    rf_coordinates,
    my_gwr,
    sarm,
    slx,
]
model_names = [
    "linear regression",
    "RF (coordinates)",
    "GWR",
    "SAR",
    "SLX",
]

datasets = ["meuse", "plants", "atlantic"]  # , "deforestation"]

np.random.seed(42)

for DATASET in datasets:
    print("\nDATASET", DATASET, "\n")

    dataset_x = {}  # per default: x
    dataset_y = {}  # per default: y
    data_path = os.path.join(
        "data_submission", "spatial_regression", DATASET + ".csv"
    )

    data = pd.read_csv(data_path)
    # print("Number of samples", len(data))
    # print(data.head())

    results = cross_validation(data)
    results.to_csv(
        os.path.join(out_path, f"results_{DATASET}_folds.csv"), index=False
    )

    results_grouped = (
        results.groupby(["Method"])
        .mean()
        .drop(["fold", "max_depth"], axis=1)
        .sort_values("RMSE")
    )
    results_grouped.to_csv(os.path.join(out_path, f"results_{DATASET}.csv"))

    print(results_grouped)
    print("--------------")
