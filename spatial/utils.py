import numpy as np


def dist_to_weight(dist):
    return 1 - (dist / dist.max())


def weighted_std(df):
    weights_normed = df["weight"] / df["weight"].sum()
    # average = np.average(values, weights=weights)
    # return np.sqrt(np.average((values - average) ** 2, weights=weights))
    average = (df["prediction"] * weights_normed).sum()
    weighted_std = np.sqrt(
        np.average((df["prediction"] - average) ** 2, weights=weights_normed)
    )
    return weighted_std


def weighted_avg(df):
    return (df["prediction"] * df["weight"] / df["weight"].sum()).sum()
