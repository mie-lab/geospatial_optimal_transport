import numpy as np
import argparse
import collections


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
    parser.add_argument("-m", "--model", default="linear", type=str)
    parser.add_argument("--multi_vs_ind", default="multi", type=str)
    parser.add_argument("-r", "--reconcile", default=0, type=int)
    parser.add_argument("-x", "--hierarchy", default=0, type=int)
    parser.add_argument("-l", "--lags", default=5, type=int)
    parser.add_argument("--output_chunk_length", default=1, type=int)
    parser.add_argument("--n_epochs", default=50, type=int)
    parser.add_argument("--num_stacks", default=3, type=int)
    # set to 0 for not using past covariates
    parser.add_argument("--lags_past_covariates", default=0, type=int)
    args = parser.parse_args()
    if args.reconcile > 0:
        assert args.hierarchy != 0
    return args


def construct_name(args):
    arg_dict = vars(args)
    arg_dict.pop("data_path")
    arg_dict.pop("station_path")
    arg_dict.pop("out_path")
    sorted_dict = collections.OrderedDict(sorted(arg_dict.items()))
    args_as_str = "_".join([str(v) for v in sorted_dict.values()])
    print("Saving as ", args_as_str)
    return args_as_str


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
