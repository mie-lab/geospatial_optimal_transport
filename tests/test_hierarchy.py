import numpy as np
from geoemd.utils import spacetime_cost_matrix


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


def test_spacetime_cost_matrix():
    nr_stations = 10
    # design pairwise distance matrix with at most 10km distance
    dist_matrix = np.random.rand(nr_stations, nr_stations) * 10
    final_cost_matrix = spacetime_cost_matrix(dist_matrix)
    print(final_cost_matrix.shape)

    x, y = (5, 25)
    s1, s2 = x % nr_stations, y % nr_stations
    t1, t2 = x // nr_stations, y // nr_stations
    print(f"pred - station {s1} time {t1}, gt - station {s2} time {t2}")
    print(final_cost_matrix[x, y])

    # # single-cell version to compute the same value
    # final_cost_matrix_v2 = np.zeros((time_steps * nr_stations, time_steps * nr_stations))
    # for x in range(len(final_cost_matrix_v2)):
    #     for y in range(len(final_cost_matrix_v2)):
    #         # get station numbers
    #         s_pred, s_gt = x%nr_stations, y%nr_stations
    #         t_pred, t_gt = x//nr_stations, y//nr_stations
    #         relocation_time = time_matrix[s_pred, s_gt]
    #         # if the time where we are relocating to (t2) is higher, we don't have many costs, we just use demand later that we estimated for now
    #         waiting_time = (t_gt - t_pred) * forward_cost if t_gt >= t_pred else (t_pred - t_gt) * backward_cost

    #         final_cost_matrix_v2[x, y] = max([waiting_time, relocation_time])
    # #         print(x, y, waiting_time, relocation_time)


test_spacetime_cost_matrix()
