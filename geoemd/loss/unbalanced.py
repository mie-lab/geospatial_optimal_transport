import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class POTUnbalancedSinkhorn:
    def __init__(self, C, normalize_c=True, reg=0.1, reg_m=10, numItermax=200):
        if isinstance(C, np.ndarray):
            C = torch.from_numpy(C).double()
        # normalize to values betwen 0 and 1
        assert C.dim() == 2
        if normalize_c:
            C = C / torch.sum(C)
        self.cost_matrix = C.unsqueeze(0).to(device)
        self.reg = reg
        self.reg_m = reg_m
        self.numItermax = numItermax

    def __call__(self, a, b):
        batchsize = a.size()[0]
        if a.dim() == 3:
            steps_ahead = a.size()[1]
            a = a.reshape((batchsize * steps_ahead, -1))
            b = b.reshape((batchsize * steps_ahead, -1))
            batchsize = batchsize * steps_ahead

        loss_arr = self.unbalanced_ot_sinkhorn(a, b)
        # # batch-wise processing
        # loss_arr = torch.empty(batchsize)
        # for batch_num in range(batchsize):
        #     loss = self.unbalanced_ot_sinkhorn(
        #         a[batch_num],
        #         b[batch_num],
        #         method="sinkhorn",
        #         numItermax=1000,
        #         stopThr=1e-06,
        #         verbose=False,
        #     )
        #     # debugging:
        #     # with torch.autograd.detect_anomaly():
        #     #     loss.backward()
        #     # print(loss)
        #     loss_arr[batch_num] = loss
        return torch.sum(loss_arr)

    def unbalanced_ot_sinkhorn(
        self,
        a,
        b,
        stopThr=1e-6,
    ):
        r"""
        CODE ADAPTED FROM THE POT (Python Optimal Transport) LIBRARY
        https://pythonot.github.io/_modules/ot/unbalanced.html#sinkhorn_unbalanced2

        Solve the entropic regularization unbalanced optimal transport problem
        and return the OT plan
        The function solves the following optimization problem:

        .. math::
            W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
            \mathrm{reg}\cdot\Omega(\gamma) +
            \mathrm{reg_m} \cdot \mathrm{KL}(\gamma \mathbf{1}, \mathbf{a}) +
            \mathrm{reg_m} \cdot \mathrm{KL}(\gamma^T \mathbf{1}, \mathbf{b})

            s.t.
                \gamma \geq 0

        where :

        - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
        - :math:`\Omega` is the entropic regularization term, :math:
        `\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
        - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
        unbalanced distributions
        - KL is the Kullback-Leibler divergence

        The algorithm used for solving the problem is the generalized
        Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10, 25]
        <references-sinkhorn-knopp-unbalanced>`


        Parameters
        ----------
        a : array-like (dim_a,)
            Unnormalized histogram of dimension `dim_a`
        b : array-like (dim_b,) or array-like (dim_b, n_hists)
            One or multiple unnormalized histograms of dimension `dim_b`
            If many, compute all the OT distances (a, b_i)
        M : array-like (dim_a, dim_b)
            loss matrix
        reg : float
            Entropy regularization term > 0
        reg_m: float
            Marginal relaxation term > 0
        numItermax : int, optional
            Max number of iterations
        stopThr : float, optional
            Stop threshold on error (> 0)
        verbose : bool, optional
            Print information along iterations
        log : bool, optional
            record log if True


        Returns
        -------
            - ot_distance : (n_hists,) array-like
                the OT distance between :math:`\mathbf{a}` and each of the
                histograms :math:`\mathbf{b}_i`

        Examples
        --------

        >>> import ot
        >>> a=[.5, .5]
        >>> b=[.5, .5]
        >>> M=[[0., 1.],[1., 0.]]
        >>> ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M, 1., 1.)
        array([[0.51122823, 0.18807035],
            [0.18807035, 0.51122823]])


        .. _references-sinkhorn-knopp-unbalanced:
        References
        ----------
        .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
            Scaling algorithms for unbalanced transport problems. arXiv
            preprint arXiv:1607.05816.

        .. [25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. :
            Learning with a Wasserstein Loss,  Advances in Neural Information
            Processing Systems (NIPS) 2015

        See Also
        --------
        ot.lp.emd : Unregularized OT
        ot.optim.cg : General regularized OT

        """
        assert len(a) > 0 and len(b) > 0
        # add 1 dim to both
        b = b.unsqueeze(-1)
        a = a.unsqueeze(-1)

        if torch.any(a < 0):
            # TODO: do I need the outputs to use softplus as well?
            # is it okay to use this if condition?
            act = torch.nn.Softplus()  # torch.nn.ReLU()  #
            a = act(a)  # + 0.01

        # nx = torch
        # nx = get_backend(self.cost_matrix, a, b)

        if self.cost_matrix.dim() == 3:
            assert a.dim() == 3 and b.dim() == 3
            batchsize, dim_a, dim_b = self.cost_matrix.shape
        else:
            batchsize = 0
            dim_a, dim_b = self.cost_matrix.shape
        # we assume that no distances are null except those of the diagonal of
        # distances
        # These are like the weights!! simply same weight for every entry
        u = torch.ones(a.size(), dtype=a.dtype) / dim_a
        v = torch.ones(b.size(), dtype=b.dtype) / dim_b

        K = torch.exp(self.cost_matrix / (-self.reg))

        fi = self.reg_m / (self.reg_m + self.reg)

        err = 1.0

        for i in range(self.numItermax):  # usually around 600 iterations
            uprev = u
            vprev = v

            Kv = torch.matmul(K, v)
            u = (a / Kv) ** fi
            if batchsize == 0:
                Ktu = torch.matmul(K.T, u)
            else:
                Ktu = torch.matmul(K.transpose(1, 2), u)
            v = (b / Ktu) ** fi

            # this is only done if we have a high numItermax
            # (otherwise it never happens anyways)
            if self.numItermax > 200:
                err_u = torch.max(torch.abs(u - uprev)) / torch.maximum(
                    torch.maximum(
                        torch.max(torch.abs(u)), torch.max(torch.abs(uprev))
                    ),
                    torch.tensor([1]),
                )
                err_v = torch.max(torch.abs(v - vprev)) / torch.maximum(
                    torch.maximum(
                        torch.max(torch.abs(v)), torch.max(torch.abs(vprev))
                    ),
                    torch.tensor([1]),
                )
                err = 0.5 * (err_u + err_v)

                if err < stopThr:
                    break

        # # to return the OT matrix:
        # return u[:, None] * K * v[None, :]
        # # to return just the distance
        if batchsize == 0:
            res = torch.einsum("ik,ij,jk,ij->k", u, K, v, self.cost_matrix)
        else:
            res = torch.einsum("ijl,ijk,ikl,ijk->il", u, K, v, self.cost_matrix)
        return res


class DeprecatedSinkhornUnbalanced:
    def __init__(
        self,
        C,
        spatiotemporal=False,
        normalize_c=True,
        reg=0.1,
        reg_m=10,
        max_iters=100,
    ):
        self.spatiotemporal = spatiotemporal
        self.reg = reg
        self.reg_m = reg_m
        self.max_iters = max_iters
        if isinstance(C, np.ndarray):
            C = torch.from_numpy(C)
        # normalize to values betwen 0 and 1
        if normalize_c:
            C = C / torch.sum(C)
        self.cost_matrix = C.to(device)

    def __call__(self, a, b):
        if self.spatiotemporal:
            # then reshape -> flatten space-time axes
            a = a.reshape((a.size()[0], -1))
            b = b.reshape((b.size()[0], -1))
        # manually add up losses for the batch
        batchsize = a.size()[0]
        for batch_sample in range(batchsize):
            if a.dim() > 2:
                # define empty array
                steps_ahead = a.size()[1]
                loss = torch.empty((batchsize, steps_ahead))
                # not spatiotemporal loss, but instead average over time axis
                for time_sample in range(steps_ahead):
                    # with torch.autograd.detect_anomaly():
                    #     loss = self.compute_emd_single(
                    #         a[batch_sample, time_sample],
                    #         b[batch_sample, time_sample],
                    #     )
                    #     loss.backward()
                    loss[batch_sample, time_sample] = self.compute_emd_single(
                        a[batch_sample, time_sample],
                        b[batch_sample, time_sample],
                    )
            else:
                loss = torch.empty(batchsize)
                loss[batch_sample] = self.compute_emd_single(
                    a[batch_sample], b[batch_sample]
                )
        # print("MEAN", torch.mean(loss))
        return torch.mean(loss)

    def compute_emd_single(self, a, b):
        emd_loss = ot.sinkhorn_unbalanced2(
            torch.clamp(a, min=0),
            b,
            self.cost_matrix,
            self.reg,
            self.reg_m,
            method="sinkhorn",
            numItermax=self.max_iters,
            stopThr=1e-06,
            verbose=False,
            log=False,
        )
        return emd_loss


if __name__ == "__main__":
    a = torch.from_numpy(
        np.array([[4, 2, 3], [4, 2, 3]])
    ).float()  # predicted distribution
    a.requires_grad = True
    b = torch.from_numpy(
        np.array([[5, 1, 3], [5, 1, 3]])
    ).float()  # new distribution
    M = torch.from_numpy(np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])).float()
    M = M / torch.sum(M)  # need to normalize in the same way to compare
    reg = 0.1
    reg_m = 10

    unb = UnbalancedSinkhorn(M, reg=reg, reg_m=reg_m)
    check = unb(a, b)
    print("My solution (adapted)", check)
    # print(check)
    # with torch.autograd.detect_anomaly():
    #     check.backward()

    import ot

    check = ot.sinkhorn_unbalanced2(
        a[0],
        b[0],
        M,
        reg,
        reg_m,
        method="sinkhorn",
        numItermax=1000,
        stopThr=1e-06,
        verbose=False,
        log=False,
    )
    print("POT solution", check * len(a))
