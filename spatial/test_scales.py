import os
import numpy as np
import pandas as pd
import argparse
import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from augment import *
from utils import dist_to_weight, weighted_std, weighted_avg


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


dataset_target = {
    "plants": "richness_species_vascular",
    "meuse": "zinc",
    "atlantic": "Rate",
    "deforestation": "deforestation_quantile",
    "california_housing": "median_house_value",
}
FOLDS = 10
dist_cutoff = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="../../z_inactive_projects/spatial_random_forest/data/",
    )
    parser.add_argument("-o", "--out_path", type=str, default="outputs")
    parser.add_argument("-k", "--k_neighbors", type=int, default=5)
    parser.add_argument("--distance_band", action="store_true")
    args = parser.parse_args()

    data_path = args.data
    out_path = args.out_path
    nr_neighbors = args.k_neighbors

    os.makedirs(out_path, exist_ok=True)

    for ds in dataset_target.keys():
        print("--------", ds)
        # load data
        data = pd.read_csv(os.path.join(data_path, f"{ds}.csv"))
        target = dataset_target[ds]
        feat_cols = list(data.drop([target, "x", "y"], axis=1).columns)

        # get quantile distance as the cutoff;
        if args.distance_band:
            dist_cutoff = quantile_dist_to_nn(data, k_nn=1, quantile=0.9)
            print("Augmenting with KNN distance band ", round(dist_cutoff, 2))

        # start final dataframe
        out_df = []

        # split into train and test
        train_inds, test_inds = get_folds(len(data), nr_folds=FOLDS)

        for fold in range(FOLDS):
            train_set_orig = data.iloc[train_inds[fold]]
            test_data = data.iloc[test_inds[fold]]

            augmented_train_data = augment_data(
                train_set_orig,
                train_set_orig,
                nr_neighbors,
                dist_cutoff=dist_cutoff,
            )

            # remove duplicates
            augmented_train_data.drop_duplicates(inplace=True)
            if fold == 0:
                print(
                    "increased train data size by",
                    len(augmented_train_data) / len(train_set_orig),
                )

            # baseline model
            model = RandomForestRegressor(n_estimators=100)
            model.fit(train_set_orig[feat_cols], train_set_orig[target])

            # train on augmented data
            model_aug = RandomForestRegressor(n_estimators=100)
            model_aug.fit(
                augmented_train_data[feat_cols], augmented_train_data[target]
            )

            # augment the test data as well
            augmented_test_data = augment_data(
                test_data, data, nr_neighbors, dist_cutoff=dist_cutoff
            )
            augmented_test_data["weight"] = dist_to_weight(
                augmented_test_data["dist"]
            )

            # # Make predictions with different methods
            # 1)  use basic model and predict basic data
            pred_basic = model.predict(test_data[feat_cols])
            # 2) use model trained with augmentation and predict basic data
            pred_trainaug = model_aug.predict(test_data[feat_cols])

            # 3) use model trained with augmentation and predict augmented data
            augmented_test_data["prediction"] = model_aug.predict(
                augmented_test_data[feat_cols]
            )
            pred_trainaug_testaug = augmented_test_data.groupby("orig").agg(
                {"prediction": "mean"}
            )
            unc_trainaug_testaug = augmented_test_data.groupby("orig").agg(
                {"prediction": "std"}
            )
            # weighted average and uncertainty
            pred_trainaug_testaug_weighted = augmented_test_data.groupby(
                "orig"
            ).apply(weighted_avg)
            unc_trainaug_testaug_weighted = augmented_test_data.groupby(
                "orig"
            ).apply(weighted_std)

            # 4) use basic model and predict augmented data
            augmented_test_data["prediction"] = model.predict(
                augmented_test_data[feat_cols]
            )
            pred_testaug = augmented_test_data.groupby("orig").agg(
                {"prediction": "mean"}
            )
            unc_testaug = augmented_test_data.groupby("orig").agg(
                {"prediction": "std"}
            )
            pred_testaug_weighted = augmented_test_data.groupby("orig").apply(
                weighted_avg
            )
            unc_testaug_weighted = augmented_test_data.groupby("orig").apply(
                weighted_std
            )

            # collect results:
            res_df = pd.DataFrame()
            res_df["gt"] = test_data[target]

            res_df["pred_basic"] = pred_basic
            res_df["pred_trainaug"] = pred_trainaug
            res_df["pred_testaug"] = pred_testaug.values
            res_df["pred_trainaug_testaug"] = pred_trainaug_testaug.values

            # add uncertainty estimates
            res_df["unc_testaug"] = unc_testaug.values
            res_df["unc_trainaug_testaug"] = unc_trainaug_testaug.values

            # add weighted measures:
            res_df["pred_testaug_weighted"] = pred_testaug_weighted.values
            res_df["unc_testaug_weighted"] = unc_testaug_weighted.values
            res_df[
                "pred_trainaug_testaug_weighted"
            ] = pred_trainaug_testaug_weighted.values
            res_df[
                "unc_trainaug_testaug_weighted"
            ] = unc_trainaug_testaug_weighted.values

            # add errors
            for pred_col in [col for col in res_df.columns if "pred_" in col]:
                res_df["MSE_" + pred_col[5:]] = (
                    res_df["gt"] - res_df[pred_col]
                ) ** 2
                res_df["MAE_" + pred_col[5:]] = abs(
                    res_df["gt"] - res_df[pred_col]
                )

            # add fold
            res_df["fold"] = fold
            res_df["test_inds"] = test_inds[fold]

            out_df.append(res_df)

        out_df = pd.concat(out_df)
        out_df.to_csv(os.path.join(out_path, ds + "_res.csv"), index=False)

        print("RMSE basic", np.sqrt(out_df["MSE_basic"].mean()))
        print(
            "RMSE train augmented and test augmented",
            np.sqrt(out_df["MSE_trainaug_testaug"].mean()),
        )

