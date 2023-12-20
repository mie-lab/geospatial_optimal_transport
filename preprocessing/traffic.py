import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams.update({"font.size": 15})

"""
Data taken from here: https://github.com/MengzhangLI/STFGNN/tree/master/data/PEMS08
"""

in_path_npz = os.path.join("data", "traffic", "PEMS08.npz")
in_path_adj = os.path.join("data", "traffic", "PEMS08.csv")
out_path = os.path.join("data", "traffic")

# read raw data
data = np.load(in_path_npz)
data = data["data"]
(
    time_series_len,
    nr_sensors,
    channels,
) = (
    data.shape
)  # 170 censors, we have the cost between them, so remains what is the 3?
# select only traffic flow (second channel is ocucpancy rate, third is speed)
flow_values = data[:, :, 0]
# add timeslot column
base = pd.to_datetime("2016-07-01 00:00:00")
time_index = pd.date_range(
    start=base, end=None, periods=len(flow_values), freq="5min"
)
traffic_flow = pd.DataFrame(
    flow_values.astype(int),
    columns=np.arange(flow_values.shape[1]),
    index=time_index,
)
traffic_flow.index.name = "timeslot"
traffic_flow.to_csv(os.path.join(out_path, "data.csv"))

# read station adjacency matrix
data_csv = pd.read_csv(in_path_adj)
# make graph and cost matrix
G = nx.from_pandas_edgelist(
    data_csv,
    source="from",
    target="to",
    edge_attr="cost",
    create_using=nx.Graph,
)
sp = nx.floyd_warshall(G, weight="cost")

sp_matrix = np.zeros((nr_sensors, nr_sensors))
for i in range(nr_sensors):
    for j in range(nr_sensors):
        sp_matrix[i, j] = sp[i][j]

cost_matrix = pd.DataFrame(
    sp_matrix,
    index=np.arange(flow_values.shape[1]),
    columns=np.arange(flow_values.shape[1]),
)
cost_matrix.index.name = "station_id"
cost_matrix.to_csv(os.path.join(out_path, "cost.csv"))

# plt.imshow(sp_matrix)
# plt.colorbar()
