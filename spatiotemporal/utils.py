import numpy as np


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
