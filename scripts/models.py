import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sprf.spatial_random_forest import SpatialRandomForest
from sprf.geographical_random_forest import GeographicalRandomForest
from scipy.spatial import distance_matrix
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from pykrige.rk import RegressionKriging
import spreg
import libpysal


def linear_regression(train_data, test_data, feat_cols=[], **kwargs):
    regr = LinearRegression()
    regr.fit(train_data[feat_cols], train_data["label"])
    test_pred = regr.predict(test_data[feat_cols])
    return test_pred


def rf_coordinates(
    train_data, test_data, feat_cols=[], max_depth=30, n_estim=100, **kwargs
):
    regr = RandomForestRegressor(n_estimators=n_estim, max_depth=max_depth)
    regr.fit(
        train_data[["x_coord", "y_coord"] + feat_cols], train_data["label"]
    )
    test_pred = regr.predict(test_data[["x_coord", "y_coord"] + feat_cols])
    return test_pred


def rf_global(
    train_data, test_data, feat_cols=[], max_depth=30, n_estim=100, **kwargs
):
    regr = RandomForestRegressor(n_estimators=n_estim, max_depth=max_depth)
    regr.fit(train_data[feat_cols], train_data["label"])
    test_pred = regr.predict(test_data[feat_cols])
    return test_pred


def get_weights_as_array(points, max_points):
    dist_matrix = distance_matrix(points, points)
    my_w = 1 / dist_matrix
    my_w[my_w == np.inf] = 0
    sorted_vals_points = np.sort(my_w, axis=1)[:, -max_points]
    my_w[my_w < np.expand_dims(sorted_vals_points, 1)] = 0
    my_w = my_w / np.expand_dims(np.sum(my_w, axis=1), 1)
    return my_w


def morans_i(y, w):
    sum_numerator = 0
    sum_denominator = 0
    normed_y = y - np.mean(y)
    for i in range(len(w)):
        for j in range(len(w)):
            sum_numerator += w[i, j] * normed_y[i] * normed_y[j]
        sum_denominator += normed_y[i] ** 2
    return (len(y) / np.sum(w)) * (sum_numerator / sum_denominator)


def slx(train_data, test_data, w_cutoff=20, feat_cols=[], **kwargs):
    divide_test = len(train_data)
    together = pd.concat((train_data, test_data))
    w = get_weights_as_array(together[["x_coord", "y_coord"]].values, w_cutoff)
    X = together[feat_cols].values
    lagged_X = np.matmul(w, X)
    X_with_lag = np.hstack((X, lagged_X))

    regr = LinearRegression()
    # fit training with lagged X
    regr.fit(X_with_lag[:divide_test], train_data["label"].values)
    # predict test part
    test_pred = regr.predict(X_with_lag[divide_test:])
    return test_pred


def sarm(train_data, test_data, feat_cols=[], **kwargs):
    X = train_data[feat_cols].values
    Y = train_data["label"].values
    # try:
    if True:
        dist_with_next = (
            train_data[["x_coord", "y_coord"]]
            - train_data[["x_coord", "y_coord"]].shift(1)
        ) ** 2
        thresh = np.sqrt(
            dist_with_next["x_coord"] + dist_with_next["y_coord"]
        ).median()
        w = libpysal.weights.DistanceBand(
            train_data[["x_coord", "y_coord"]].values.astype(float),
            threshold=thresh,
            binary=False,
        )
        model = spreg.GM_Lag(Y, X, w=w)
        # print("pseudo r2", model.pr2)
        intercept = model.betas[0]
        coeff = model.betas[1:-1]
        roh = model.betas[-1]
        # basic is just X\beta
        test_pred_basic = (
            np.matmul(test_data[feat_cols].values, coeff) + intercept
        )

        # complex is with the second part
        def get_weights_as_array(points, max_dist):
            dist_matrix = distance_matrix(points, points)
            my_w = 1 / dist_matrix
            my_w[my_w == np.inf] = 0
            my_w[my_w < 1 / max_dist] = 0
            return my_w

        W = get_weights_as_array(
            test_data[["x_coord", "y_coord"]].values, thresh
        )
        test_pred = np.matmul(
            np.linalg.inv(np.identity(len(W)) - roh * W), test_pred_basic
        )
    # except:
    #     print("ERROR in SAR")
    #     test_pred = np.zeros(len(test_data)) + np.mean(Y)
    return test_pred


def my_gwr(train_data, test_data, feat_cols=[], **kwargs):
    try:
        train_coords = np.array(train_data[["x_coord", "y_coord"]])
        train_y = np.expand_dims(train_data["label"].values, 1)
        train_x = np.array(train_data[feat_cols])
        # bandwidth selection
        gwr_selector = Sel_BW(
            train_coords, train_y, train_x, fixed=True, kernel="exponential"
        )
        gwr_bw = gwr_selector.search(criterion="AICc")
        # create and train model
        gwr_model = GWR(
            train_coords,
            train_y,
            train_x,
            gwr_bw,
            kernel="exponential",
            fixed=True,
        )
        gwr_results = gwr_model.fit()

        test_coords = np.array(test_data[["x_coord", "y_coord"]])
        test_x = np.array(test_data[feat_cols])
        # predict
        test_pred = gwr_model.predict(
            test_coords, test_x, gwr_results.scale, gwr_results.resid_response
        ).predictions
        return test_pred
    except:
        print("GWR not possible")
        return np.zeros(len(test_data))
