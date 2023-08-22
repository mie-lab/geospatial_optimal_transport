import numpy as np
import argparse
import collections
from geoemd.config import CONFIG


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
    parser.add_argument("-m", "--model", default="linear", type=str)
    parser.add_argument("--multi_vs_ind", default="multi", type=str)
    parser.add_argument("-r", "--reconcile", default=0, type=int)
    parser.add_argument("-x", "--hierarchy", default=0, type=int)
    parser.add_argument("-l", "--lags", default=24, type=int)
    parser.add_argument("--output_chunk_length", default=3, type=int)
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


def get_dataset_name(in_path_data):
    if "bikes" in in_path_data and "2015" in in_path_data:
        return "bikes_2015"
    elif "bikes" in in_path_data:
        return "bikes"
    elif "charging" in in_path_data:
        return "charging"
    elif "carsharing" in in_path_data:
        return "carsharing"
    else:
        raise ValueError("In path wrong, does not match available dataset")
