import numpy as np
import torch
from geoemd.loss.sinkhorn_loss import SinkhornLoss
import wasserstein


class InterpretableUnbalancedOT:
    def __init__(
        self,
        cost_matrix,
        normalize_c=True,
        penalty_unb="max",
        compute_exact=False,
        norm_sum_1=False,
        spatiotemporal=False,
    ):
        """
        Initialize unbalanced OT class with cost matrix
        Arguments:
            norm_sum_1: Whether to compute the Wasserstein distance between
                distributions or the actual value range
        """
        self.compute_exact = compute_exact
        self.norm_sum_1 = norm_sum_1
        if penalty_unb == "max":
            penalty_unb = np.max(cost_matrix)
        clen = len(cost_matrix)
        extended_cost_matrix = np.zeros((clen + 1, clen + 1))
        extended_cost_matrix[:clen, :clen] = cost_matrix
        extended_cost_matrix[clen, :] = penalty_unb
        extended_cost_matrix[:, clen] = penalty_unb

        if normalize_c:
            extended_cost_matrix = extended_cost_matrix / np.max(
                extended_cost_matrix
            )
        if compute_exact:
            self.cost_matrix = extended_cost_matrix
            self.balancedOT = wasserstein.EMD()
        else:
            self.balancedOT = SinkhornLoss(
                extended_cost_matrix,
                blur=0.1,
                reach=0.01,
                scaling=0.1,
                mode="unbalanced",
                spatiotemporal=spatiotemporal,
            )

    def __call__(self, a, b):
        # compute mass that has to be imported or exported
        diff = (torch.sum(a, dim=-1) - torch.sum(b, dim=-1)).unsqueeze(-1)

        diff_pos = torch.relu(diff)
        diff_neg = torch.relu(diff * -1)

        extended_a = torch.cat((a, diff_neg), dim=-1)
        extended_b = torch.cat((b, diff_pos), dim=-1)

        if self.compute_exact:
            assert a.size()[0] == 1 and b.size()[0] == 1 and a.dim() == 2
            a_np = extended_a.squeeze().detach().numpy().astype(float)
            b_np = extended_b.squeeze().detach().numpy().astype(float)
            if self.norm_sum_1:
                a_np = a_np / np.sum(a_np)
                b_np = b_np / np.sum(b_np)
            else:
                # still need this code to avoid numeric errors
                a_np = a_np / np.sum(a_np) * np.sum(b_np)
            cost = self.balancedOT(a_np, b_np, self.cost_matrix)
            return cost
        else:
            return self.balancedOT(extended_a, extended_b)


if __name__ == "__main__":
    test_cdist = np.array(
        [
            [0.0, 0.9166617229649182, 0.8011636143804466, 1.0],
            [0.9166617229649182, 0.0, 0.2901671214052399, 0.5131642591866252],
            [0.8011636143804466, 0.2901671214052399, 0.0, 0.28166962442054133],
            [1.0, 0.5131642591866252, 0.28166962442054133, 0.0],
        ]
    )
    ot_obj = InterpretableUnbalancedOT(test_cdist, compute_exact=True)
    print(
        ot_obj(
            # torch.tensor([[1, 3, 2, 4], [1, 3, 2, 4]]),
            # torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]),
            torch.tensor([[1, 3, 2, 4]]),
            torch.tensor([[1, 2, 3, 4]]),
        )
    )
