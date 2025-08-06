import numpy as np
from scipy.spatial.transform import Rotation
import tensorly as tl
from tulca import TULCA

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

tab10 = {
    0: "#4E79A7",  # blue
    1: "#F28E2B",  # orange
    2: "#59A14F",  # green
    3: "#E15759",  # red
    4: "#B07AA1",  # purple
    5: "#9C755F",  # brown
    6: "#FF9DA7",  # pink
    7: "#EDC948",  # yellow-ish
    8: "#76B7B2",  # teal
    9: "#BAB0AC",  # gray
}

#
# 1. Generate synthetic data
#
y = np.array([0] * 400 + [1] * 200)
X = np.zeros((len(y), 10, 3))
for t in [0, 1, 2, 3, 7, 8, 9]:
    np.random.seed(0)
    X[:, t, :] = np.random.rand(len(y), 3)
X = X * 20.0 - 10.0


blob1 = np.random.normal(
    loc=[-2.5, -3.0, 0.0], scale=0.75, size=(int(np.sum(y == 0) / 2), 3)
)
blob2 = np.random.normal(
    loc=[-2.5, 3.0, 0.0], scale=0.75, size=(int(np.sum(y == 0) / 2), 3)
)
blob3 = np.random.normal(
    loc=[2.5, 2.5, 0.0], scale=[0.5, 4.0, 0.5], size=(np.sum(y == 1), 3)
)
blobs = np.vstack((blob1, blob2, blob3))
for t in range(4, 7):
    X[y == 0, t, :] = blobs[y == 0]
    X[y == 1, t, :] = blobs[y == 1]
R = Rotation.from_euler("z", 170, degrees=True).as_matrix()
X = X @ R.T

# plot synthetic data
vis_time = [0, 4, 9]
cmap = LinearSegmentedColormap.from_list("", [tab10[1], tab10[2]])
fig = plt.figure(figsize=(5 * len(vis_time), 5))
mins = []
maxs = []
for idx in vis_time:
    mins.append(X[:, idx, :].min(axis=0))
    maxs.append(X[:, idx, :].max(axis=0))
mins = np.array(mins).min(axis=0)
maxs = np.array(maxs).max(axis=0)
ranges = maxs - mins
max_range = ranges.max()
lims = np.array(
    [(maxs + mins - max_range * 1.02) / 2, (maxs + mins + max_range * 1.02) / 2]
).T

for i, idx in enumerate(vis_time):
    ax = fig.add_subplot(1, len(vis_time), i + 1, projection="3d")
    ax.set_zticks([])
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.scatter(*X[:, idx, :].T, c=y, s=20, lw=0.2, edgecolors="#888888", cmap=cmap)
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_zlim(lims[2])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("1st variable")
    ax.set_ylabel("2nd variable")
    ax.set_zlabel("3rd variable")
plt.tight_layout()
plt.show()

#
# 2. Comparison of decomposition methods
#
results = {}

(core, factors), rec_errors = tl.decomposition.partial_tucker(
    X, modes=[1, 2], rank=[3, 2]
)
cp_weights, cp_factors = tl.decomposition.parafac(core, rank=2)
results["Tucker"] = {}
results["Tucker"]["Ms"] = factors
results["Tucker"]["cp_weights"] = cp_weights
results["Tucker"]["cp_factors"] = cp_factors

weights = {
    "T-PCA": {"w_tg": [1.0, 1.0], "w_bg": [0.0, 0.0], "w_bw": [0.0, 0.0]},
    "T-LDA": {"w_tg": [0.0, 0.0], "w_bg": [1.0, 1.0], "w_bw": [1.0, 1.0]},
    "T-cPCA": {"w_tg": [1.0, 0.0], "w_bg": [0.0, 1.0], "w_bw": [0.0, 0.0]},
    "TULCA": {"w_tg": [1.0, 0.0], "w_bg": [0.0, 1.0], "w_bw": [1.0, 1.0]},
}

for decomp_type in weights.keys():
    results[decomp_type] = {}
    tulca = TULCA(
        n_components=[3, 2],
        w_tg=weights[decomp_type]["w_tg"],
        w_bg=weights[decomp_type]["w_bg"],
        w_bw=weights[decomp_type]["w_bw"],
        optimization_method="evd",
    )
    tulca.fit(X, y)
    results[decomp_type]["Ms"] = tulca.Ms_
    Z = tulca.transform(X)

    # CP decompositon
    cp_weights, cp_factors = tl.decomposition.parafac(Z, rank=2)
    results[decomp_type]["cp_weights"] = cp_weights
    results[decomp_type]["cp_factors"] = cp_factors

fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(3 * 5, 3), tight_layout=True)
for i, (decomp_type, result) in enumerate(results.items()):
    axs[i].set_title(decomp_type)
    axs[i].scatter(
        result["cp_factors"][0][:, 0],
        result["cp_factors"][0][:, 1],
        c=y,
        alpha=0.8,
        cmap=LinearSegmentedColormap.from_list("", [tab10[1], tab10[2]]),
    )
    axs[i].set_title(decomp_type)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
plt.show()

fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(3 * 5, 1 * 4), tight_layout=True)
for i, (decomp_type, result) in enumerate(results.items()):
    axs[0, i].bar(
        np.arange(len(result["cp_factors"][1][:, 0])),
        result["cp_factors"][1][:, 0],
        color="gray",
    )
    axs[1, i].bar(
        np.arange(len(result["cp_factors"][1][:, 1])),
        result["cp_factors"][1][:, 1],
        color="gray",
    )
    axs[2, i].bar(
        np.arange(len(result["cp_factors"][2][:, 0])),
        result["cp_factors"][2][:, 0],
        color="gray",
    )
    axs[3, i].bar(
        np.arange(len(result["cp_factors"][2][:, 1])),
        result["cp_factors"][2][:, 1],
        color="gray",
    )
    axs[0, i].set_xlabel("Time component")
    axs[0, i].set_ylabel("x-axis")
    axs[1, i].set_xlabel("Time component")
    axs[1, i].set_ylabel("y-axis")
    axs[2, i].set_xlabel("Variable component")
    axs[2, i].set_ylabel("x-axis")
    axs[3, i].set_xlabel("Variable component")
    axs[3, i].set_ylabel("y-axis")
    for j in range(4):
        axs[j, i].hlines(
            y=0,
            xmin=-0.5,
            xmax=len(result["cp_factors"][j // 2 + 1][:, 0]) - 0.5,
            linewidth=1,
            color="#444444",
        )
        axs[j, i].set_xticks([])
        axs[j, i].set_yticks([])
        axs[j, i].set(frame_on=False)
plt.show()

fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(4 * 5, 2), tight_layout=True)
for i, (decomp_type, result) in enumerate(results.items()):
    axs[i].set_title(decomp_type)
    axs[i].imshow(
        result["Ms"][0],
        cmap="RdBu_r",
        aspect="auto",
        origin="lower",
        vmin=-np.abs(result["Ms"][0]).max(),
        vmax=np.abs(result["Ms"][0]).max(),
        extent=[0.5, 3.5, 0.5, 10.5],
    )
    axs[i].set_title(decomp_type)
    axs[i].set_xticks([1, 2, 3])
    axs[i].set_yticks([1, 4, 7, 10])
    axs[i].set_xlabel("Time component")
    axs[i].set_ylabel("Time")
plt.show()

fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(4 * 5, 1.5), tight_layout=True)
for i, (decomp_type, result) in enumerate(results.items()):
    axs[i].set_title(decomp_type)
    axs[i].imshow(
        result["Ms"][1],
        cmap="RdBu_r",
        aspect="auto",
        vmin=-np.abs(result["Ms"][1]).max(),
        vmax=np.abs(result["Ms"][1]).max(),
        origin="lower",
        extent=[0.5, 2.5, 0.5, 3.5],
    )
    axs[i].set_title(decomp_type)
    axs[i].set_xticks([1, 2])
    axs[i].set_yticks([1, 2, 3])
    axs[i].set_xlabel("Variable component")
    axs[i].set_ylabel("Variable")
plt.show()
