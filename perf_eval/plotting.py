import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Table 2
df = pd.read_csv("./result/perf_eval.csv")
print(df.groupby("data").mean())

# Figure
df = pd.read_csv("./result/perf_eval_mhealth_synthetic.csv")
df = df.groupby(["n_times", "n_insts"]).mean().reset_index()
plt.figure(figsize=(4, 2.5))
plt.hlines(y=0.01, xmin=0, xmax=6000, linewidth=0.8, color="#cccccc")
plt.hlines(y=0.1, xmin=0, xmax=6000, linewidth=0.8, color="#cccccc")
plt.hlines(y=1, xmin=0, xmax=6000, linewidth=0.8, color="#cccccc")
plt.hlines(y=10, xmin=0, xmax=6000, linewidth=0.8, color="#cccccc")
plt.hlines(y=100, xmin=0, xmax=6000, linewidth=0.8, color="#cccccc")
plt.hlines(y=1000, xmin=0, xmax=6000, linewidth=0.8, color="#cccccc")
# plt.vlines(x=5000, ymin=0.001, ymax=8000, linewidth=0.8, color="#cccccc")
for n_times in [100, 1000, 10000]:
    plt.plot(
        df[df["n_times"] == n_times]["n_insts"],
        df[df["n_times"] == n_times]["TULCA-all"],
        label=rf"$K_1={n_times}$",
    )
plt.xlabel(r"$K_2$")
plt.ylabel("Completion time (sec)")
plt.xscale("log")
plt.yscale("log")
plt.xlim([8, 5500])
plt.ylim([0.002556779, 9000])
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("tulca_all.pdf")
plt.show()
