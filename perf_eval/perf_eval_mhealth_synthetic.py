import os
import numpy as np
from tulca import TULCA

import time

# Fig. 10
if not os.path.exists("./result/"):
    os.makedirs("./result")
outfile_path = f"./result/perf_eval_mhealth_synthetic.csv"
with open(outfile_path, "w") as f:
    f.write(f"n_times,n_insts,trial,TULCA-all,TULCA-update\n")

n_trials = 10
# synthetic data generation from MHELATH
for n_times in [100, 1000, 10000]:
    y = np.load(f"../data/mhealth_synthetic/time_labels_{n_times}.npy")
    for n_insts in [
        10,
        20,
        40,
        60,
        80,
        100,
        200,
        400,
        600,
        800,
        1000,
        2000,
        3000,
        5000,
    ]:
        X = np.load(f"../data/mhealth_synthetic/tensor_{n_times}x10x23.npy")
        # copy if n_insts needed to be larger than 10
        X = np.repeat(X, n_insts // X.shape[1], axis=1)
        X = X[:, :n_insts, :]

        for i in range(n_trials):
            start_time = time.time()
            tulca = TULCA(
                n_components=[3, 3],
                w_tg=np.random.rand(len(np.unique(y))),
                w_bg=np.random.rand(len(np.unique(y))),
                w_bw=np.random.rand(len(np.unique(y))),
                optimization_method="evd",
                # convergence_ratio=1e-4,
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
                    f"{n_times},{n_insts},{i},{time_tulca_all},{time_tulca_update}\n"
                )

            print(f"{n_times},{n_insts},{i},{time_tulca_all},{time_tulca_update}")
