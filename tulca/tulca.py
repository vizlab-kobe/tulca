import numpy as np
import tensorly as tl

from scipy import linalg

import pymanopt
from pymanopt.manifolds import Grassmann
from pymanopt.optimizers import TrustRegions

# from factor_analyzer import Rotator


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
    """Contrastive PCA with efficient implemetation and automatic alpha selection.

    Parameters
    ----------
    n_components: int or array-like with shape(n_modes_excluding_the_first,), optional, (default=2)
        Number of components for each mode, except for the first mode.
        If None, it is set to half length of each mode.
        If a scalar is given, it is used for all modes (except for the first mode).
    w_tg: array-like, shape(n_classes,), optional (default=None)
        Target weights for within-class covariance matrices (C_wi) of .
        If None, it is set to zero vector.
        When w_bg, w_bw are also None, this is equivalent for tensor discriminant analysis.
    w_bg: array-like, shape(n_classes,), optional (default=None)
        Background weights for within-class covariance matrices (C_wi).
        If None, it is set to one vector.
    w_bw: array-like, shape(n_classes,), optional (default=None)
        Weights for between-class covariance matrices (C_bw).
        If None, it is set to one vector.
    gamma_a: float, optional (default=0)
        Regularization parameter for C_a, the sum of target within-class and between-class covariance matrices.
        If 0, no regularization is applied.
    gamma_b: float, optional (default=0)
        Regularization parameter for C_b, corresponding to background within-class covariance matrix.
        If 0, no regularization is applied.
    alphas: float or array-like with shape(n_modes_excluding_the_first,), optional (default=None)
        Alphas for each mode (except for the first mode), which is used to control the trade-off between
        C_a (the sum of target within-class and between-class covariance matrices)
        and C_b (background within-class covariance matrix).
        If a mode's alpha is None, it is selected automatically by using iterative optimization (see the ULCA [1] paper's Eq. 10 and 11).
        If a scalar is given, it is used for all modes (except for the first mode).
        If None, it is set to None for each mode.
        [1] Takanori Fujiwara, Xinhai Wei, Jian Zhao, and Kwan-Liu Ma, Interactive Dimensionality Reduction for Comparative Analysis, IEEE Transactions on Visualization and Computer Graphics, 2022.
    convergence_ratio: float, optional (default=1e-4)
        Ratio of improvement in alpha value to stop the optimization when using "evd" as an optimization method.
    max_iter: int, optional, (default=100)
        Maximum number of iterations used by the optimization when using "evd" as an optimization method.
    optimization_method: str, optional (default="evd")
        Method to use for optimization. Currently, "evd" (eigenvalue decomposition)
        and "manopt" (Pymanopt) are supported.
        If "evd", it uses eigenvalue decomposition to find the optimal projection matrix.
        If "manopt", it uses Pymanopt library to solve the optimization problem.
    manifold_generator: Pymanopt Manifold class, optional (default=Grassmann)
        Manifold class used for generating manifold optimization problem when using "manopt" as an optimization method.
    manifold_optimizer: Pymanopt Optimizer class, optional (default=TrustRegions())
        Optimizer class used for solving manifold optimization problem when using "manopt" as an optimization method.
    apply_consist_axes: bool, optional, (default=True)
        If True, signs of axes and order of axes are adjusted and generate
        more consistent axes' signs and orders.
        Refer to Sec. 4.2.4 in Fujiwara et al., Interactive Dimensionality
        Reduction for Comparative Analysis, 2021
    verbosity: int, optional, (default=0)
        Level of information logged by the solver while it operates, 0 is
        silent, 1 or greater shows the improvement process of alpha.

    Methods
    ----------
    fit: Fit the model to the data.
    fit_transform: Fit the model to the data and transform it.
    transform: Transform the data using the fitted model.
    fit_with_new_weights: Fit the model with new weights (w_tg, w_bg, w_bw).
    get_projection_matrices: Get the projection matrices for each mode.
    get_current_alphas: Get the current alphas for each mode.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import tensorly as tl
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.preprocessing import scale

    >>> from tulca import TULCA

    >>> X = np.load("./data/highschool_2012/tensor.npy")
    >>> y = np.array(pd.read_csv("./data/highschool_2012/instance_labels.csv")["label"])

    >>> # the above data has TxNxD shape and labels for instances (not for time points)
    >>> # so, move axes to make NxTxD
    >>> X = np.moveaxis(X, 1, 0)
    >>> # perform scaling
    >>> X = tl.fold(scale(tl.unfold(X, 0)), 0, X.shape)
    >>> n_components = np.array([15, 5])

    >>> # TULCA with default parameters performs tensor discriminant analysis (TDA)
    >>> tulca = TULCA(n_components=n_components)
    >>> Z = tulca.fit_transform(X, y)

    >>> # sample of getting projection matrices
    >>> Ms = tulca.get_projection_matrices()

    >>> #
    >>> # Usage example of compressed tensor
    >>> #
    >>> # Example 1. Using CP decomposition similar to the TULCA paper
    >>> cp_weights, cp_factors = tl.decomposition.parafac(Z, rank=2)

    >>> fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    >>> axs[0].set_title("CP decomp of the original tensor")
    >>> axs[0].scatter(*tl.decomposition.parafac(X, rank=2)[1][0].T, c=y, cmap="tab10")
    >>> axs[1].set_title("CP decomp of the TULCA result")
    >>> axs[1].scatter(*cp_factors[0].T, c=y, cmap="tab10")
    >>> plt.tight_layout()
    >>> plt.show()

    >>> # Example 2. Applying dimensionality reduction (DR) to n_samples unfolded tensors
    >>> from sklearn.manifold import TSNE

    >>> dr = TSNE(n_components=2)

    >>> fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    >>> axs[0].set_title("Nonlinear DR on the original data")
    >>> axs[0].scatter(*dr.fit_transform(tl.unfold(X, 0)).T, c=y, cmap="tab10")
    >>> axs[1].set_title("Nonlinear DR on the TDA result")
    >>> axs[1].scatter(*dr.fit_transform(tl.unfold(Z, 0)).T, c=y, cmap="tab10")
    >>> plt.tight_layout()
    >>> plt.show()

    >>> #
    >>> # Example updating weights
    >>> #

    >>> # Example 1. Apply different weights when TULCA is not fitted yet
    >>> # (this is slower than re-fitting with new weights)
    >>> tulca = TULCA(n_components=n_components, w_tg=[0.0, 1.0, 1.0, 1.0], w_bg=[1.0, 0.0, 0.0, 0.0], w_bw=[1.0, 1.0, 1.0, 1.0])
    >>> # NOTE: the above weights minimize only the variance of Class 0 (blue)
    >>> Z1 = tulca.fit_transform(X, y)

    >>> # Example 2. Apply different weights when TULCA is already fitted
    >>> # (this is much faster than fit)
    >>> tulca.fit_with_new_weights(w_tg=[0.0, 0.1, 1.0, 0.3], w_bg=[1.0, 0.1, 0.0, 0.0], w_bw=[1.0, 1.0, 1.0, 1.0])
    >>> Z2 = tulca.transform(X)

    >>> # Example 1. Using CP decomposition similar to the TULCA paper
    >>> fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
    >>> axs[0].set_title("CP decomp of the TULCA result (1st fit)")
    >>> axs[0].scatter(*tl.decomposition.parafac(Z, rank=2)[1][0].T, c=y, cmap="tab10")
    >>> axs[1].set_title("CP decomp of the TULCA result (2nd fit)")
    >>> scatter = axs[1].scatter(*tl.decomposition.parafac(Z1, rank=2)[1][0].T, c=y, cmap="tab10")
    >>> axs[2].set_title("CP decomp of the TULCA result (3rd fit)")
    >>> scatter = axs[2].scatter(*tl.decomposition.parafac(Z2, rank=2)[1][0].T, c=y, cmap="tab10")
    >>> plt.tight_layout()
    >>> plt.show()
    """

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
        # apply_varimax=False,
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
        # self.apply_varimax = apply_varimax
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
        """Fit the TULCA model.
        Parameters
        ----------
        X: array-like of shape(n_elements_for_mode0, n_elements_for_mode1, ..., n_elements_for_modeN)
            Input high-order tensor.
        y: array-like of shape (n_elements_for_mode0)
            Labels of mode0 of the input tensor.
        Returns
        -------
        self: TULCA instance
            Fitted TULCA instance.
        """
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
        self._optimize()

        return self

    def fit_with_new_weights(
        self, w_tg=None, w_bg=None, w_bw=None, gamma_a=None, gamma_b=None
    ):
        """
        Fit the TULCA model with new weights. This fit is much faster than the initial fit.
        Parameters
        ----------
        w_tg: array-like, shape(n_classes,), optional (default=None)
            Target weights for within-class covariance matrices (C_wi) of .
            If None, w_tg keeps the current value.
        w_bg: array-like, shape(n_classes,), optional (default=None)
            Background weights for within-class covariance matrices (C_wi).
            If None, w_bg keeps the current value.
        w_bw: array-like, shape(n_classes,), optional (default=None)
            Weights for between-class covariance matrices (C_bw).
            If None, w_bw keeps the current value.
        gamma_a: float, optional (default=0)
            Regularization parameter for C_a, the sum of target within-class and between-class covariance matrices.
            If 0, no regularization is applied.
        gamma_b: float, optional (default=0)
            Regularization parameter for C_b, corresponding to background within-class covariance matrix.
            If 0, no regularization is applied.
        Returns
        -------
        self: TULCA instance
            Fitted TULCA instance with updated weights.
        """
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

        self._optimize()

        return self

    def _optimize(self):
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

            # if self.apply_varimax and self.n_components[m] > 1:
            #     self.Ms_[m] = Rotator(method="varimax").fit_transform(self.Ms_[m])
            if self.apply_consist_axes:
                # consist sign (column sum will be pos)
                self.Ms_[m] = self.Ms_[m] * np.sign(self.Ms_[m].sum(axis=0))
                # consist order (based on max value)
                self.Ms_[m] = self.Ms_[m][:, np.argsort(-self.Ms_[m].max(axis=0))]

        return self

    def transform(self, X, y=None):
        """Transform the input tensor using the fitted TULCA model.
        Parameters
        ----------
        X: array-like of shape(n_elements_for_mode0, n_elements_for_mode1, ..., n_elements_for_modeN)
            Input high-order tensor.
        y: array-like of shape (n_elements_for_mode0), optional (default=None)
            Labels of mode0 of the input tensor. Not used in the current implementation.
        Returns
        -------
        Z: numpy array, shape(n_elements_for_mode0, n_components[0], n_components[1], ..., n_components[N-1])
            Transformed tensor with reduced dimensions.
        """
        X_compressed = X
        for mode, M in enumerate(self.Ms_):
            X_compressed = tl.tenalg.mode_dot(X_compressed, M.T, mode + 1)
        return X_compressed

    def fit_transform(self, X, y):
        """Fit the TULCA model and transform the input tensor.
        Parameters
        ----------
        X: array-like of shape(n_elements_for_mode0, n_elements_for_mode1, ..., n_elements_for_modeN)
            Input high-order tensor.
        y: array-like of shape (n_elements_for_mode0)
            Labels of mode0 of the input tensor.
        Returns
        -------
        Z: numpy array, shape(n_elements_for_mode0, n_components[0], n_components[1], ..., n_components[N-1])
            Transformed tensor with reduced dimensions.
        """
        return self.fit(X, y).transform(X, y)

    def get_projection_matrices(self):
        """Get the projection matrices for each mode.
        Returns
        -------
        Ms: numpy array, shape(n_modes_excluding_the_first, n_features, n_components)
            Projection matrices for each mode (except for the first mode).
        """
        return np.copy(self.Ms_)

    def get_current_alphas(self):
        """Get the current alphas for each mode.
        Returns
        -------
        alphas: numpy array, shape(n_modes_excluding_the_first,)
            Current alphas for each mode (except for the first mode).
        """
        return np.copy(self.alphas_)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import tensorly as tl
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
