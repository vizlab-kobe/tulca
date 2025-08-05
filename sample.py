# install matplotlib if you want to run the example with plots
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

    # TULCA with default parameters performs tensor discriminant analysis (TDA)
    tulca = TULCA(n_components=n_components)
    Z = tulca.fit_transform(X, y)

    # sample of getting projection matrices
    Ms = tulca.get_projection_matrices()

    #
    # Usage example of compressed tensor
    #
    # Example 1. Using CP decomposition similar to the TULCA paper
    cp_weights, cp_factors = tl.decomposition.parafac(Z, rank=2)

    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    axs[0].set_title("CP decomp of the original tensor")
    axs[0].scatter(*tl.decomposition.parafac(X, rank=2)[1][0].T, c=y, cmap="tab10")
    axs[1].set_title("CP decomp of the TULCA result")
    axs[1].scatter(*cp_factors[0].T, c=y, cmap="tab10")
    plt.tight_layout()
    plt.show()

    # Example 2. Applying dimensionality reduction (DR) to n_samples unfolded tensors
    from sklearn.manifold import TSNE

    dr = TSNE(n_components=2)

    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    axs[0].set_title("Nonlinear DR on the original data")
    axs[0].scatter(*dr.fit_transform(tl.unfold(X, 0)).T, c=y, cmap="tab10")
    axs[1].set_title("UMAP on the TDA result")
    axs[1].scatter(*dr.fit_transform(tl.unfold(Z, 0)).T, c=y, cmap="tab10")
    plt.tight_layout()
    plt.show()
