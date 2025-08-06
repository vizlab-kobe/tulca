import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale


# Download the MHEALTH dataset from UCI Machine Learning Repository
# (https://archive.ics.uci.edu/dataset/319/mhealth+dataset)
# and place the files in the `mhealth_row` directory.

all_data = []
all_labels = []
file_names = [f"./mhealth_row/mHealth_subject{i}.txt" for i in range(1, 11)]

data_for_all_insts = []
label_for_all_insts = []
for file_name in file_names:
    df = pd.read_csv(file_name, sep="\t", header=None)

    # remove rows that have the last column 0, 6, 7, 8, or 12
    df = df[~df.iloc[:, -1].isin([0, 6, 7, 8, 12])]
    labels = np.array(df.iloc[:, -1], dtype=int)
    data = np.array(df.iloc[:, :-1])

    data_for_all_insts.append(data)
    label_for_all_insts.append(labels)


def sampling(
    data_for_all_insts, label_for_all_insts, sampling_interval=50, aiming_n_times=None
):
    n_labels = len(np.unique(label_for_all_insts[0]))
    n_insts = len(data_for_all_insts)

    if aiming_n_times is not None:
        sampling_interval = 3000 * n_labels // aiming_n_times

    sampled_data = []
    sampled_labels = []
    for data, labels in zip(data_for_all_insts, label_for_all_insts):
        # trim into 3000 timepoints (each label have slightly more than 3000 timepoints)
        # then sample by sampling_interval
        for label in np.unique(labels):
            selected_data = data[labels == label][:3000:sampling_interval]
            if aiming_n_times is not None:
                selected_data = selected_data[: int(np.ceil(aiming_n_times / n_labels))]
            sampled_data.append(selected_data)
            sampled_labels.append(np.array([label] * selected_data.shape[0]))

    X = np.concatenate(sampled_data, axis=0)
    X = scale(X)

    # reshape to tensor
    n_times = X.shape[0] // n_insts
    X = X.reshape((n_insts, n_times, -1))
    X = np.moveaxis(X, 1, 0)

    y = np.concatenate(sampled_labels, axis=0)
    y = y.reshape((n_insts, n_times, -1))
    y = np.moveaxis(y, 1, 0)[:, 0, 0]
    # relabel the labels to be 0, 1, 2, ...
    y = pd.Categorical(y).codes

    return X, y


X, y = sampling(data_for_all_insts, label_for_all_insts, sampling_interval=50)
np.save(f"./tensor.npy", X)
np.save(f"./time_labels.npy", y)

# synthetic data generation from MHELATH
dir_path = "../mhealth_synthetic/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
for n_times in [100, 1000, 10000]:
    for n_insts in [10]:
        X, y = sampling(data_for_all_insts, label_for_all_insts, aiming_n_times=n_times)
        X = X[:n_times]
        y = y[:n_times]

        X = np.repeat(X, n_insts // X.shape[1], axis=1)
        X = X[:, :n_insts, :]

        np.save(f"{dir_path}tensor_{X.shape[0]}x{X.shape[1]}x{X.shape[2]}.npy", X)
        np.save(f"{dir_path}time_labels_{X.shape[0]}.npy", y)
