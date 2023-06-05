from station_hierarchy import StationHierarchy
from optimal_transport import OptimalTransportLoss

load_path = "outputs/test1"
gt_col, pred_col = "gt_0", "pred_linear_multi_no_0"

station_hierarchy = StationHierarchy()
station_hierarchy.init_from_file(load_path)

transport_loss = OptimalTransportLoss(station_hierarchy)
transport_loss.transport_from_centers(gt_col, pred_col)
transport_loss.transport_equal_dist(gt_col, pred_col)
