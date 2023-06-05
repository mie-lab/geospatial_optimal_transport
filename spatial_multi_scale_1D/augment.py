import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import time


def get_avg_dist_to_neighbors(data, nr_neighbors=3):
    tree = BallTree(data[["x", "y"]], leaf_size=15, metric="euclidean")
    distances, indices = tree.query(data[["x", "y"]], k=nr_neighbors + 1)
    avg_dist_nn = np.mean(distances[1:])
    return avg_dist_nn  # avg distance to three nearest neighbors


def quantile_dist_to_nn(data, k_nn=1, quantile=0.5):
    tree = BallTree(data[["x", "y"]], leaf_size=15, metric="euclidean")
    distances, indices = tree.query(data[["x", "y"]], k=k_nn + 1)
    # print(distances.shape)
    # plt.hist(distances[:, k_nn], bins=30)
    # plt.show()
    return np.quantile(distances[:, k_nn], quantile)


def augment_by_distance(train_set_orig, scale, shifts=[0, 0.35, 0.7]):
    # add other scales
    # for i, scale in enumerate(scales):
    new_data = []
    for shift_factor in shifts:
        shift = shift_factor * scale
        train_set_orig["x_grouped"] = (train_set_orig["x"] + shift) // scale
        train_set_orig["y_grouped"] = (train_set_orig["y"] + shift) // scale
        # print(
        #     "avg number of neighbors",
        #     train_set_orig.groupby(["x_grouped", "y_grouped"])
        #     .agg({"x_grouped": "count"})
        #     .mean()
        #     .values,
        # )
        excerpt = train_set_orig.groupby(["x_grouped", "y_grouped"]).mean()
        # print(len(excerpt))
        new_data.append(excerpt)
    new_data = pd.concat(new_data)
    # print(len(new_data))
    return new_data


def augment_by_neighbors(train_set_orig, all_data=None, nr_neighbors=3):
    """
    Can yield duplicates because the same set of points are paired several times
    """
    # tic = time.time()
    if all_data is None:
        all_data = train_set_orig
    tree = BallTree(all_data[["x", "y"]], leaf_size=15, metric="euclidean")
    _, indices = tree.query(train_set_orig[["x", "y"]], k=nr_neighbors + 1)
    new_rows = []
    for r in indices:
        new_rows.append(all_data.iloc[r].mean())
    new_df = pd.DataFrame(new_rows)
    # print("t:", time.time() - tic)
    return new_df


def augment_by_neighbors_efficient(
    data_to_augment, all_data=None, nr_neighbors=2
):
    # tic = time.time()
    """efficient by using join operations instead of iterating over rows"""
    # copy input data - want to augment, not to replace
    query_data = data_to_augment.copy().reset_index().drop(["index"], axis=1)
    # if no other data is given, we use the query data
    if all_data is None:
        all_data = query_data

    tree = BallTree(all_data[["x", "y"]], leaf_size=15, metric="euclidean")
    distances, indices = tree.query(query_data[["x", "y"]], k=nr_neighbors + 1)

    assert all(all_data.index == range(0, len(all_data)))

    for n in range(nr_neighbors):
        query_data[f"neigh_{n}"] = indices[:, n + 1]
        query_data = query_data.merge(
            all_data,
            how="left",
            right_index=True,
            left_on=f"neigh_{n}",
            suffixes=("", f"_neigh_{n}"),
        )

    # put the distance in a column for a potential weighting
    query_data["avg_dist"] = np.mean(distances, axis=1)

    # merge the columns
    for f in all_data.columns:
        f_related_cols = [f + f"_neigh_{n}" for n in range(nr_neighbors)]
        query_data[f] = query_data[f_related_cols].mean(axis=1)
    # drop the neighbor columns
    query_data.drop(
        [c for c in query_data.columns if "neigh_" in c], axis=1, inplace=True
    )
    # print("t2", time.time() - tic)
    return query_data


def augment_data(
    data_to_augment, all_data=None, nr_neighbors=4, dist_cutoff=None
):
    # set index to range index -> needed for tree search
    test_data = data_to_augment.reset_index(drop=True).drop(
        ["traintest"], axis=1, errors="ignore"
    )
    if all_data is None:
        all_data = data_to_augment
    all_data = all_data.reset_index(drop=True).drop(
        ["traintest"], axis=1, errors="ignore"
    )

    test_data["orig"] = test_data.index  # original to the current index
    test_data["dist"] = 0  # the raw sample has zero distance to itself
    test_data["k_neighbor"] = 0

    # query in tree
    tree = BallTree(all_data[["x", "y"]], leaf_size=15, metric="euclidean")
    distances, indices = tree.query(test_data[["x", "y"]], k=nr_neighbors + 1)

    # collect augmented data
    augmented = [test_data]
    for n in range(nr_neighbors):
        neighbor_values = all_data.iloc[indices[:, n + 1]].reset_index(
            drop=True
        )
        avg_2 = (neighbor_values + test_data) / 2
        avg_2["orig"] = test_data.index
        avg_2["k_neighbor"] = n + 1
        avg_2["dist"] = distances[:, n + 1]
        if dist_cutoff is not None:
            # print(len(avg_2))
            avg_2 = avg_2[avg_2["dist"] < dist_cutoff]
            # print("Reducing due to distance cutoff", len(avg_2))
        augmented.append(avg_2)
    augmented = pd.concat(augmented)
    return augmented
