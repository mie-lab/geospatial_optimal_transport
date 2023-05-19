import geomloss
import torch
import numpy as np
from torch.nn import MSELoss

device = "cuda" if torch.cuda.is_available() else "cpu"


class SinkhornLoss:
    def __init__(self, C, normalize_c=True, blur=0.5, **sinkhorn_kwargs):
        if isinstance(C, np.ndarray):
            C = torch.from_numpy(C)
        # normalize to values betwen 0 and 1
        if normalize_c:
            C = C / torch.sum(C)
        if C.dim() != 3:
            if C.dim() != 2:
                raise ValueError("cost matrix C must have 2 or 3 dimensions")
            C = C.unsqueeze(0)

        # cost matrics and locs both need a static representation and are
        # modified later to match the batch size
        self.cost_matrix = C.to(device)
        self.cost_matrix_original = self.cost_matrix.clone()
        self.dummy_locs = torch.tensor(
            [[[i] for i in range(C.size()[-1])]]
        ).float()
        self.dummy_locs_orig = self.dummy_locs.clone()

        # sinkhorn loss
        self.loss_object = geomloss.SamplesLoss(
            loss="sinkhorn",
            cost=self.get_cost,
            backend="tensorized",
            debias=True,
            blur=blur,
            **sinkhorn_kwargs,
        )

    def get_cost(self, a, b):
        return self.cost_matrix

    def __call__(self, a_in, b_in):
        # Adapt cost matrix size to the batch size
        batch_size = a_in.size()[0]
        if self.cost_matrix.size()[0] != batch_size:
            self.cost_matrix = self.cost_matrix_original.repeat(
                (batch_size, 1, 1)
            )
            self.dummy_locs = self.dummy_locs_orig.repeat((batch_size, 1, 1))

        # normalize a and b
        adim = a_in.dim() - 1
        a = a_in / torch.sum(a_in, axis=adim).unsqueeze(adim)
        b = b_in / torch.sum(b_in, axis=adim).unsqueeze(adim)

        # check if we predicted several steps ahead
        steps_ahead = a.size()[1]
        if a.dim() > 2 and steps_ahead > 1:
            result = torch.empty((steps_ahead, batch_size))
            for i in range(steps_ahead):
                result[i] = self.loss_object(
                    a[:, i], self.dummy_locs, b[:, i], self.dummy_locs
                )
            loss = torch.mean(result, dim=0)
        else:
            loss = self.loss_object(a, self.dummy_locs, b, self.dummy_locs)
        return torch.sum(loss)


class DistributionMSE:
    def __init__(self) -> None:
        self.standard_mse = MSELoss()

    def __call__(self, a_in, b_in):
        # normalize a and b
        adim = a_in.dim() - 1
        a = a_in / torch.sum(a_in, axis=adim).unsqueeze(adim)
        b = b_in / torch.sum(b_in, axis=adim).unsqueeze(adim)

        # apply standard MSE
        return self.standard_mse(a, b)


def sinkhorn_loss_from_numpy(a, b, cost_matrix, sinkhorn_kwargs={}):
    a = torch.tensor(a.tolist()).float()
    b = torch.tensor(b.tolist()).float()
    # cost_matrix = torch.tensor([cost_matrix])
    # # Testing for the case where multiple steps ahead are predicted
    # a = a.unsqueeze(1).repeat(1, 3, 1)
    # b = b.unsqueeze(1).repeat(1, 3, 1)
    # print("Before initializing", cost_matrix.shape, a.size(), b.size())
    loss = SinkhornLoss(cost_matrix, **sinkhorn_kwargs)
    return loss(a, b)


if __name__ == "__main__":
    test_cdist = np.array(
        [
            [0.0, 0.9166617229649182, 0.8011636143804466, 1.0],
            [0.9166617229649182, 0.0, 0.2901671214052399, 0.5131642591866252],
            [0.8011636143804466, 0.2901671214052399, 0.0, 0.28166962442054133],
            [1.0, 0.5131642591866252, 0.28166962442054133, 0.0],
        ]
    )
    print(
        sinkhorn_loss_from_numpy(
            np.array([[1, 3, 2, 4], [1, 3, 2, 4]]),
            np.array([[1, 2, 3, 4], [1, 2, 3, 4]]),
            test_cdist,
        )
    )
