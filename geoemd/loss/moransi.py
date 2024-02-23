from typing import Any
import torch
import numpy as np
from torch.nn import MSELoss

device = "cuda" if torch.cuda.is_available() else "cpu"


class MoransiLoss:
    def __init__(
        self,
        C,
        normalize_c=True,
        weight_matrix_method="minus",
        spatiotemporal=False,
        apply_sqrt=True,
        **kwargs,
    ):
        """
        Implements the numerator of Moran's I (without normalization by variance or by W)
        NOTE: This only works for quadratic C
        if pred and gt have the same shape
        TODO: loss can be negative in this way!
        """
        assert C.shape[0] == C.shape[1], "C must be quadratic for MoransI"
        self.spatiotemporal = spatiotemporal
        # adapt cost matrix type
        if isinstance(C, np.ndarray):
            C = torch.from_numpy(C)

        # construct weight matric:
        if normalize_c:
            # normalize even before computing W -> leading to higher weights
            C = C / torch.max(C)

        if weight_matrix_method == "ratio":
            C = C.fill_diagonal_(1)  # needed to avoid zero division
            if apply_sqrt:
                W = 1 / torch.sqrt(
                    C
                )  # actual weight construction w_ij = 1/c_ij
            else:
                W = 1 / C
            W = W.fill_diagonal_(0)
            # normalize to values betwen 0 and 1
            if normalize_c:
                W = W / torch.max(W)
        elif weight_matrix_method == "minus":
            W = C * (-1)
        else:
            raise NotImplementedError(
                "weight_matrix_method must be in {minus, ratio}"
            )

        # adapt size to batch dimension
        if W.dim() != 3:
            if W.dim() != 2:
                raise ValueError("cost matrix C must have 2 or 3 dimensions")
            W = W.unsqueeze(0)

        # weight matrix to device
        self.W = W.to(device)

    def __call__(self, pred: torch.tensor, target: torch.tensor) -> float:
        # Flatten one axis -> either for spatiotemporal OT or treating the
        # temporal axis as batch
        batch_size = pred.size()[0]
        if self.spatiotemporal:
            # flatten space-time axes
            pred = pred.reshape((batch_size, -1))
            target = target.reshape((batch_size, -1))
        elif target.dim() == 3:
            # if we have to flatten at all, flatten time over the batch size
            steps_ahead = pred.size()[1]
            pred = pred.reshape((batch_size * steps_ahead, -1))
            target = target.reshape((batch_size * steps_ahead, -1))

        # compute residuals - now just 2 dimensions, first is batch size
        residuals = pred - target

        # use matmul to create outer producet (pred_i - gt_i) * (pred_j - gt_j)
        residuals_1 = residuals.unsqueeze(2)
        residuals_2 = residuals.unsqueeze(1)
        outer_product = torch.matmul(residuals_1, residuals_2)

        # multiply by c_ij and sum over i, j
        return torch.mean(self.W * outer_product)


class MoransiCombinedLoss:
    def __init__(self, C, spatiotemporal=False) -> None:
        self.standard_mse = MSELoss()
        self.moransi_error = MoransiLoss(C, spatiotemporal=spatiotemporal)
        self.dist_weight = 1

    def __call__(self, a_in, b_in):
        mse_loss = self.standard_mse(a_in, b_in)
        moransi_loss = self.moransi_error(a_in, b_in)
        # for checking calibration of weighting
        # print(mse_loss, self.dist_weight * moransi_loss)
        return mse_loss + self.dist_weight * moransi_loss
