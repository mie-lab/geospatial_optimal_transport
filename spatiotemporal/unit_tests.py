import numpy as np


def test_hierarchy(hier, demand_agg, station_groups, stations_locations):
    example_group = "Group_28"
    # step 1: assert that the hierarchy is consistent with the aggregated demand
    child_1, child_2 = hier[example_group]

    assert all(
        demand_agg[child_1] + demand_agg[child_2] == demand_agg[example_group]
    )
    # assert that the coordinates stations are close
    print(
        "distance of stations",
        np.linalg.norm(
            station_groups.loc[child_1, ["x", "y"]].values
            - station_groups.loc[child_2, ["x", "y"]].values
        ),
    )
    collect_stations = []

    def get_children(group):
        if "Group" not in group:
            collect_stations.append(group)
        else:
            for child in hier[group]:
                get_children(child)

    get_children(example_group)
    print("GET CHILDREN", collect_stations)

    # assert coordinates are correct
    assert np.isclose(
        station_groups.loc[example_group].values
        == stations_locations.loc[np.array(collect_stations).astype(int)]
        .mean()
        .values,
        0,
    )
