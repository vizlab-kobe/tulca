import numpy as np
import tensorly as tl

from scipy import linalg
from factor_analyzer import Rotator

import pymanopt
from pymanopt.manifolds import Grassmann
from pymanopt.optimizers import TrustRegions


def _generate_covs(X, y):
    classes = np.unique(y)
    modes = np.arange(X.ndim - 1)  # exclude sample mode
    n_samples = X.shape[0]
    n_classes = len(classes)
    n_modes = len(modes)

    # generate each sample's flatterning matrix for each mode
    matrices = np.empty(n_modes, dtype=object)
    for m in modes:
        # this is the same as np.array([tl.unfold(X[row], m) for row in range(n_samples)])
        # m + 1 is because the first axis corresponds to samples
        matrices[m] = np.swapaxes(
            tl.unfold(X, m + 1).reshape(
                X.shape[m + 1], n_samples, int(X.size / X.shape[m + 1] / n_samples)
            ),
            0,
            1,
        )

    # n_sample matrices for each class and each mode
    matrices_by_class = np.empty((n_classes, n_modes), dtype=object)
    for c in classes:
        for m in modes:
            matrices_by_class[c, m] = matrices[m][y == c]

    #
    # compute within and between covariances for each class and each mode
    #
    means = np.empty(n_modes, dtype=object)
    for m in modes:
        means[m] = matrices[m].mean(axis=0)

    means_by_class = np.empty((n_classes, n_modes), dtype=object)
    for c in classes:
        for m in modes:
            means_by_class[c, m] = matrices_by_class[c, m].mean(axis=0)

    # within-class covariance matrices (C_wi for each class and each mode)
    # TODO: find simpler naming for Cwis and Cbws
    Cwis_by_class = np.empty((n_classes, n_modes), dtype=object)
    for c in classes:
        for m in modes:
            _mats = matrices_by_class[c, m] - means_by_class[c, m]
            # this is the same as Cwis[c, m] = np.sum([mat @ mat.T for mat in mats])
            Cwis_by_class[c, m] = np.matmul(_mats, np.swapaxes(_mats, 1, 2)).sum(axis=0)

    # between-class covariance matrices (C_bw for each class and each mode)
    Cbws_by_class = np.empty((n_classes, n_modes), dtype=object)
    for c in classes:
        n_class_samples = np.sum(y == c)
        for m in modes:
            _means = means_by_class[c, m] - means[m]
            Cbws_by_class[c, m] = n_class_samples * _means @ _means.T

    return Cwis_by_class, Cbws_by_class


def _combine_covs(
    Cwis_by_class,
    Cbws_by_class,
    w_tg=None,
    w_bg=None,
    w_bw=None,
    gamma_a=0,
    gamma_b=0,
):
    n_classes, n_modes = Cwis_by_class.shape

    # default: Tensor LDA
    if w_tg is None:
        w_tg = np.zeros(n_classes)
    if w_bg is None:
        w_bg = np.ones(n_classes)
    if w_bw is None:
        w_bw = np.ones(n_classes)

    Cwi_tgs = np.sum([Cwis_by_class[c] * w_tg[c] for c in range(n_classes)], axis=0)
    Cwi_bgs = np.sum([Cwis_by_class[c] * w_bg[c] for c in range(n_classes)], axis=0)
    Cbws = np.sum([Cbws_by_class[c] * w_bw[c] for c in range(n_classes)], axis=0)

    # C_a and C_b for each mode
    Cas = np.empty(n_modes, dtype=object)
    Cbs = np.empty(n_modes, dtype=object)
    # see Eq. 6-8 in ULCA paper
    for m in range(n_modes):
        Cas[m] = Cwi_tgs[m] + Cbws[m] + gamma_a * np.eye(*Cwi_tgs[m].shape)
        Cbs[m] = Cwi_bgs[m] + gamma_b * np.eye(*Cwi_bgs[m].shape)

    return Cas, Cbs


def gen_cost_tulca(manifold, C_a, C_b, alpha):
    @pymanopt.function.autograd(manifold)
    def cost(M):
        return np.trace(M.T @ C_b @ M) / np.trace(M.T @ C_a @ M)

    @pymanopt.function.autograd(manifold)
    def cost_with_alpha(M):
        return np.trace(M.T @ (alpha * C_b - C_a) @ M)

    if alpha:
        return cost_with_alpha
    else:
        return cost


class TULCA:
    def __init__(
        self,
        n_components=None,
        w_tg=None,
        w_bg=None,
        w_bw=None,
        gamma_a=0,
        gamma_b=0,
        alphas=None,
        convergence_ratio=1e-4,
        max_iterations=100,
        optimization_method="evd",
        manifold_generator=Grassmann,
        manifold_optimizer=TrustRegions(),
        apply_varimax=False,
        apply_consist_axes=True,
        verbosity=False,
    ):
        # NOTE: currently, gamma_a, gamma_b is only one value across modes
        # but, can easily change them to be one value for each mode by slightly changing the code
        # (similarly, for w_tg, w_bg, w_bw)
        self.n_components = n_components
        self.w_tg = w_tg
        self.w_bg = w_bg
        self.w_bw = w_bw
        self.gamma_a = gamma_a
        self.gamma_b = gamma_b
        self.alphas = alphas
        self.convergence_ratio = convergence_ratio
        self.max_iterations = max_iterations
        self.optimization_method = optimization_method
        self.manifold_generator = manifold_generator
        self.manifold_optimizer = manifold_optimizer
        self.apply_varimax = apply_varimax
        self.apply_consist_axes = apply_consist_axes
        self.verbosity = verbosity

    def _apply_evd(self, C_a, C_b, alpha, n_components):
        C = C_a - alpha * C_b
        schur_form, v = linalg.schur(C)
        w = linalg.eigvals(schur_form)
        top_eigen_indices = np.argsort(-w)
        return v[:, top_eigen_indices[:n_components]]

    def _optimize_with_evd(self, C_a, C_b, alpha, n_components):
        if alpha is not None:
            M = self._apply_evd(C_a, C_b, alpha, n_components)
        else:
            # initial solutions with alpha=0
            alpha = 0
            M = self._apply_evd(C_a, C_b, alpha, n_components)

            improved_ratio = 1
            for _ in range(self.max_iterations):
                prev_alpha = alpha
                alpha = np.trace(M.T @ C_a @ M) / np.trace(M.T @ C_b @ M)
                M = self._apply_evd(C_a, C_b, alpha, n_components)

                improved_ratio = np.abs(prev_alpha - alpha) / alpha
                if self.verbosity > 0:
                    print(f"alpha: {alpha}, improved: {improved_ratio}")
                if improved_ratio < self.convergence_ratio:
                    break

        return M, alpha

    def _optimize_with_manopt(self, C_a, C_b, alpha, n_components):
        mode_length = C_a.shape[0]

        manifold = self.manifold_generator(mode_length, n_components)
        problem = pymanopt.Problem(manifold, gen_cost_tulca(manifold, C_a, C_b, alpha))
        self.manifold_optimizer._verbosity = self.verbosity
        M = self.manifold_optimizer.run(problem).point

        if alpha is None:
            alpha = 1 / problem.cost(M)

        return M, alpha

    def fit(self, X, y):
        modes = np.arange(X.ndim - 1)  # exclude sample mode
        n_modes = len(modes)
        self.alphas_ = self.alphas
        self.Ms_ = np.empty(n_modes, dtype=object)

        if self.alphas_ is None:
            self.alphas_ = np.array([None] * n_modes)
        elif np.isscalar(self.alphas_):
            self.alphas_ = np.array([self.alphas_] * n_modes)

        if self.n_components is None:
            # set half length of each mode
            self.n_components = (np.array(X.shape[1:]) / 2).astype(int)
        elif np.isscalar(self.n_components):
            self.n_components = np.array([self.n_components] * n_modes)

        # prepare covariances
        self.Cwis_by_class_, self.Cbws_by_class_ = _generate_covs(X, y)
        # perform optimization
        self.optimize()

        return self

    def fit_with_new_weights(
        self, w_tg=None, w_bg=None, w_bw=None, gamma_a=None, gamma_b=None
    ):
        if w_tg is not None:
            self.w_tg = w_tg
        if w_bg is not None:
            self.w_bg = w_bg
        if w_bw is not None:
            self.w_bw = w_bw
        if gamma_a is not None:
            self.gamma_a = gamma_a
        if gamma_b is not None:
            self.gamma_b = gamma_b

        self.optimize()

        return self

    def optimize(self):
        C_as, C_bs = _combine_covs(
            self.Cwis_by_class_,
            self.Cbws_by_class_,
            self.w_tg,
            self.w_bg,
            self.w_bw,
            self.gamma_a,
            self.gamma_b,
        )

        n_modes = len(C_as)
        for m in range(n_modes):
            C_a = C_as[m] if np.any(C_as[m]) else np.eye(*C_as[m].shape)
            C_b = C_bs[m] if np.any(C_bs[m]) else np.eye(*C_bs[m].shape)

            if self.optimization_method == "evd":
                M, alpha = self._optimize_with_evd(
                    C_a, C_b, self.alphas_[m], self.n_components[m]
                )
            else:
                M, alpha = self._optimize_with_manopt(
                    C_a, C_b, self.alphas_[m], self.n_components[m]
                )
            self.Ms_[m] = M
            self.alphas_[m] = alpha

            if self.apply_varimax and self.n_components[m] > 1:
                self.Ms_[m] = Rotator(method="varimax").fit_transform(self.Ms_[m])
            if self.apply_consist_axes:
                # consist sign (column sum will be pos)
                self.Ms_[m] = self.Ms_[m] * np.sign(self.Ms_[m].sum(axis=0))
                # consist order (based on max value)
                self.Ms_[m] = self.Ms_[m][:, np.argsort(-self.Ms_[m].max(axis=0))]

        return self

    def transform(self, X, y=None):
        X_compressed = X
        for mode, M in enumerate(self.Ms_):
            X_compressed = tl.tenalg.mode_dot(X_compressed, M.T, mode + 1)
        return X_compressed

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)

    def get_projection_matrices(self):
        return np.copy(self.Ms_)

    def get_current_alphas(self):
        return np.copy(self.alphas_)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import tensorly as tl
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import scale

    from tulca import TULCA

    X = np.load("./data/highschool_2012/tensor.npy")
    y = np.array(pd.read_csv("./data/highschool_2012/instance_labels.csv")["label"])

    # the above data has TxNxD shape and labels for instances (not for time points)
    # so, move axes to make NxTxD
    X = np.moveaxis(X, 1, 0)
    # perform scaling
    X = tl.fold(scale(tl.unfold(X, 0)), 0, X.shape)
    n_components = np.array([15, 5])

    # TULCA with default parameters performs tensor discriminant analysis
    tulca = TULCA(n_components=n_components)
    Z = tulca.fit_transform(X, y)

    # usage example of compressed tensor
    # (applying DR to n_samples unfolded tensors)
    # from umap import UMAP

    # dr = UMAP(n_components=2)

    # fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    # axs[0].scatter(*dr.fit_transform(tl.unfold(X, 0)).T, c=y, cmap="tab10")
    # axs[0].set_title("UMAP on the original data")
    # axs[1].scatter(*dr.fit_transform(tl.unfold(Z, 0)).T, c=y, cmap="tab10")
    # axs[1].set_title("UMAP on the TDA result")
    # plt.tight_layout()
    # plt.show()
