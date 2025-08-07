import os
import numpy as np
import pandas as pd
import tensorly as tl
from tulca import TULCA

import time

# Table 2
datasets = {}

X = np.load("../data/air_quality/tensor.npy")
# create two groups from dates
dates = pd.to_datetime(pd.read_csv("../data/air_quality/times.csv")["check_time"])
y = np.zeros(len(dates), dtype=int)
y[(dates >= "2018-05-01") & (dates < "2018-10-01")] = 1
datasets["air_quality"] = {"X": X, "y": y}

X = np.load("../data/highschool_2012/tensor.npy")
# highschool dataset has labels for instances, not for time points
X = np.moveaxis(X, 1, 0)
y = np.array(pd.read_csv("../data/highschool_2012/instance_labels.csv")["label"])
datasets["highschool_2012"] = {"X": X, "y": y}

X = np.load("../data/mhealth/tensor.npy")
y = np.load("../data/mhealth/time_labels.npy")
datasets["mhealth"] = {"X": X, "y": y}

# NOTE: K log dataset is not publicly available
# X = np.load("../data/k_log/tensor.npy")
# y = np.load("../data/k_log/time_labels.npy")
# datasets["k_log"] = {"X": X, "y": y}

if not os.path.exists("./result/"):
    os.makedirs("./result")
outfile_path = f"./result/perf_eval.csv"

with open(outfile_path, "w") as f:
    f.write(f"data,trial,Tucker,TULCA-all,TULCA-update\n")

n_trials = 10
for dataset_name, dataset in datasets.items():
    X = dataset["X"]
    y = dataset["y"]

    for i in range(n_trials):
        start_time = time.time()
        tl.decomposition.partial_tucker(X, modes=[1, 2], rank=[3, 3])
        time_tucker = time.time() - start_time

        start_time = time.time()
        tulca = TULCA(
            n_components=[3, 3],
            w_tg=np.random.rand(len(np.unique(y))),
            w_bg=np.random.rand(len(np.unique(y))),
            w_bw=np.random.rand(len(np.unique(y))),
            optimization_method="evd",
            convergence_ratio=1e-4,
        )
        tulca.fit_transform(X, y)
        time_tulca_all = time.time() - start_time

        start_time = time.time()
        tulca.fit_with_new_weights(
            w_tg=np.random.rand(len(np.unique(y))),
            w_bg=np.random.rand(len(np.unique(y))),
            w_bw=np.random.rand(len(np.unique(y))),
        )
        tulca.transform(X)
        time_tulca_update = time.time() - start_time

        with open(outfile_path, "a") as f:
            f.write(
                f"{dataset_name},{i},{time_tucker},{time_tulca_all},{time_tulca_update}\n"
            )
