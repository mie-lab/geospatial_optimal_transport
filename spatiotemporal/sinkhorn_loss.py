import geomloss
import torch
import numpy as np


class SinkhornLoss:
    def __init__(self, C, n_elements, blur=0.5, **sinkhorn_kwargs):
        self.cost_matrix = C
        self.loss_object = geomloss.SamplesLoss(
            loss="sinkhorn",
            cost=self.get_cost,
            backend="tensorized",
            debias=True,
            blur=blur,
            **sinkhorn_kwargs,
        )
        self.dummy_locs = torch.tensor(
            [[[i] for i in range(n_elements)]]
        ).float()

    def get_cost(self, a, b):
        return self.cost_matrix

    def __call__(self, a, b):
        # expand batch dim
        batch_size = a.size()[0]
        if batch_size != 1:
            dummy_locs = self.dummy_locs.repeat(batch_size)
        else:
            dummy_locs = self.dummy_locs
        return self.loss_object(a, dummy_locs, b, dummy_locs).item()


def sinkhorn_loss_from_numpy(a, b, cost_matrix_numpy, sinkhorn_kwargs={}):
    nr_samples_here = len(a)

    a = torch.tensor([a.tolist()]).float()
    a = a / torch.sum(a)
    b = torch.tensor([b.tolist()]).float()
    b = b / torch.sum(b)
    cost_matrix_tensor = torch.tensor([cost_matrix_numpy])
    loss = SinkhornLoss(cost_matrix_tensor, nr_samples_here, **sinkhorn_kwargs)
    return loss(a, b)


if __name__ == "__main__":
    test_cdist = [
        [0.0, 0.9166617229649182, 0.8011636143804466, 1.0],
        [0.9166617229649182, 0.0, 0.2901671214052399, 0.5131642591866252],
        [0.8011636143804466, 0.2901671214052399, 0.0, 0.28166962442054133],
        [1.0, 0.5131642591866252, 0.28166962442054133, 0.0],
    ]
    print(
        sinkhorn_loss_from_numpy(
            np.array([1, 3, 2, 4]), np.array([1, 2, 3, 4]), test_cdist
        )
    )
