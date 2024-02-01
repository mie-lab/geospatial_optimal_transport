import numpy as np
import argparse
import collections
from scipy.spatial.distance import cdist
import wasserstein

from geoemd.loss.sinkhorn_loss import (
    CombinedLoss,
    SinkhornLoss,
)
from geoemd.loss.interpretable_unbalanced_ot import InterpretableUnbalancedOT
from geoemd.config import CONFIG, QUADRATIC_TIME, STEPS_AHEAD, FREQUENCY


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default="../data/bikes_montreal/tune_pickup.csv",
    )
    parser.add_argument(
        "-s",
        "--station_path",
        type=str,
        default="../data/bikes_montreal/tune_stations.csv",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="outputs/test",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="One of 'bikes', 'charging', 'carsharing'",
    )
    parser.add_argument("-m", "--model", default="lightgbm", type=str)
    parser.add_argument("--multi_vs_ind", default="multi", type=str)
    parser.add_argument("-r", "--reconcile", default=0, type=int)
    parser.add_argument("-x", "--hierarchy", default=0, type=int)
    parser.add_argument("-l", "--lags", default=24, type=int)
    parser.add_argument("--output_chunk_length", default=10, type=int)
    parser.add_argument("--n_epochs", default=70, type=int)
    parser.add_argument("--x_loss_function", default="basic", type=str)
    parser.add_argument("--x_scale", default=1, type=int)
    parser.add_argument("--num_stacks", default=3, type=int)
    # set to 0 for not using past covariates
    parser.add_argument("--lags_past_covariates", default=1, type=int)
    parser.add_argument("--y_clustermethod", default=None, type=str)
    parser.add_argument("--y_cluster_k", type=int, default=10)
    parser.add_argument("--model_path", type=str, default="trained_models")
    parser.add_argument("--load_model_name", type=str, default=None)
    parser.add_argument("--ordered_samples", action="store_true")
    parser.add_argument("--optimize_optuna", action="store_true")
    args = parser.parse_args()
    if args.reconcile > 0:
        assert args.hierarchy != 0
    return args


def construct_name(args):
    arg_dict = vars(args)
    # if there is a config, we overwrite all other arguments
    if args.config is not None:
        print("Warning: Replacing all training arguments with config")
        arg_dict.update(CONFIG[args.config])
    # generate save name from the sorted smaller dictionary (wo paths)
    arg_dict_wo_path = arg_dict.copy()
    for path_arg in [
        "data_path",
        "station_path",
        "out_path",
        "load_model_name",
        "model_path",
        "ordered_samples",
        "optimize_optuna",
        "config",
    ]:
        arg_dict_wo_path.pop(path_arg)
    sorted_dict = collections.OrderedDict(sorted(arg_dict_wo_path.items()))
    args_as_str = "_".join([str(v) for v in sorted_dict.values()])
    print("Saving as ", args_as_str)
    # add model name to original dictionary
    arg_dict["model_name"] = args_as_str
    return args_as_str, arg_dict


def get_error_group_level(pred, val, station_groups):
    groups_per_num = (
        station_groups.reset_index().groupby("nr_stations").agg({"group": list})
    )

    def get_err(stations_on_level):
        return np.mean(
            np.abs(
                val.pd_dataframe()[stations_on_level].values
                - pred.pd_dataframe()[stations_on_level].values
            )
        )

    errors = groups_per_num["group"].apply(get_err)
    return np.array([groups_per_num.index, errors]).swapaxes(1, 0)


def get_children_hierarchy(group, hier, levels_down):
    # wrapper function to avoid passing hier through all recursions
    def get_children(group, levels_down):
        if "Group" not in group or levels_down == 0:
            return [group]
        else:
            children_list = []
            for child in hier[group]:
                children_list.extend(get_children(child, levels_down - 1))
            return children_list

    return get_children(group, levels_down)


def create_groups_with_pred(pred, val, step_ahead, station_groups):
    gt_onestep = val[step_ahead].pd_dataframe().swapaxes(1, 0).reset_index()
    gt_onestep = gt_onestep.rename(columns={gt_onestep.columns[-1]: "gt"})
    pred_onestep = pred[step_ahead].pd_dataframe().swapaxes(1, 0).reset_index()
    pred_onestep = pred_onestep.rename(
        columns={pred_onestep.columns[-1]: "pred"}
    )
    groups_with_pred = (
        station_groups.reset_index()
        .merge(gt_onestep, how="left", left_on="group", right_on="component")
        .drop("component", axis=1)
    )
    groups_with_pred = (
        groups_with_pred.merge(
            pred_onestep, how="left", left_on="group", right_on="component"
        )
        .drop("component", axis=1)
        .set_index("group")
    )
    return groups_with_pred


def get_dataset_name(in_path_data: str) -> str:
    if "bikes" in in_path_data and "2015" in in_path_data:
        return "bikes_2015"
    elif "bikes" in in_path_data:
        return "bikes"
    elif "charging" in in_path_data:
        return "charging"
    elif "carsharing" in in_path_data:
        return "carsharing"
    elif "traffic" in in_path_data:
        return "traffic"
    else:
        raise ValueError("In path wrong, does not match available dataset")


def get_emd_loss_function(loss_fn_argument: str, time_dist_matrix: np.ndarray):
    # make spatiotemporal cost matrix
    if "temporal" in loss_fn_argument:
        spatiotemporal_cost = spacetime_cost_matrix(
            time_dist_matrix,
            time_steps=STEPS_AHEAD,
        )
    if loss_fn_argument == "emdbalancedspatial":
        return CombinedLoss(time_dist_matrix, mode="balancedSoftmax")
    elif loss_fn_argument == "emdbalancedspatiotemporal":
        # actually combined sinkhorn temporal
        return CombinedLoss(
            spatiotemporal_cost, spatiotemporal=True, mode="balancedSoftmax"
        )
    elif loss_fn_argument == "emdinterpretablespatial":
        return InterpretableUnbalancedOT(
            time_dist_matrix,
            spatiotemporal=False,
            penalty_unb=np.quantile(time_dist_matrix, 0.1),
        )
    elif loss_fn_argument == "emdunbalancedspatial":
        return SinkhornLoss(
            time_dist_matrix, mode="unbalanced", spatiotemporal=False
        )
        # training_kwargs["pl_trainer_kwargs"] = {"gradient_clip_val": 1}
    elif loss_fn_argument == "emdunbalancedspatiotemporal":
        return SinkhornLoss(
            spatiotemporal_cost, mode="unbalanced", spatiotemporal=True
        )
    else:
        raise NotImplementedError(
            "Must be emdbalancedspatial, emdunbalancedspatial,\
                    emdunbalancedspatiotemporalor emdbalancedspatiotemporal"
        )


def space_cost_matrix(
    coords1,
    coords2=None,
    speed_factor=None,
    quadratic=False,
    quadratic_factor=QUADRATIC_TIME,
):
    """
    coords: spatial coordinates (projected, distances in m)
    speed_factor: relocation speed of users (in km/h)
    """
    if coords2 is None:
        coords2 = coords1
    dist_matrix = cdist(coords1, coords2)

    # convert space to time (in h)
    if speed_factor is not None:
        time_matrix = (dist_matrix / 1000) / speed_factor
    else:
        time_matrix = dist_matrix

    # convert to perceived time
    if quadratic:
        time_matrix = (time_matrix / quadratic_factor) ** 2
    return time_matrix


def spacetime_cost_matrix(
    time_matrix,
    time_steps=3,
    forward_cost=0,
    backward_cost=1,
):
    """
    Design a space-time cost matrix that quantifies the cost across space and time
    Cell i,j is the cost from timeslot=i//nr_stations and station=i%nr_stations
    to timeslot=j//nr_stations and station=j%nr_stations

    dist_matrix: pairwise distances in m
    forward_cost: cost for using demand that was originally allocated for the
    preceding timestep (usually low) - in hours
    backward_cost: cost for using demand that was allocated for the next timestep - in hours
    """
    nr_stations = len(time_matrix)

    final_cost_matrix = np.zeros(
        (time_steps * nr_stations, time_steps * nr_stations)
    )
    for t_pred in range(time_steps):
        for t_gt in range(time_steps):
            start_x, end_x = (t_pred * nr_stations, (t_pred + 1) * nr_stations)
            start_y, end_y = (t_gt * nr_stations, (t_gt + 1) * nr_stations)
            if t_pred > t_gt:
                waiting_time = (t_pred - t_gt) * backward_cost
                final_cost_matrix[start_x:end_x, start_y:end_y] = np.maximum(
                    time_matrix, waiting_time * np.ones(time_matrix.shape)
                )
            else:
                waiting_time = (t_gt - t_pred) * forward_cost
                final_cost_matrix[start_x:end_x, start_y:end_y] = np.maximum(
                    time_matrix, waiting_time * np.ones(time_matrix.shape)
                )
    return final_cost_matrix


def balanced_ot_with_unbalanced_data(pred_vals, real_vals, dist_matrix):
    dist_matrix_normed = dist_matrix / np.max(dist_matrix)

    pred_vals_normed = pred_vals / np.sum(pred_vals) * np.sum(real_vals)
    was = wasserstein.EMD()
    emd1 = was(pred_vals_normed, real_vals, dist_matrix_normed)

    # other way round
    real_vals_normed = real_vals / np.sum(real_vals) * np.sum(pred_vals)
    was = wasserstein.EMD()
    emd2 = was(pred_vals, real_vals_normed, dist_matrix_normed)
    return emd1, emd2
