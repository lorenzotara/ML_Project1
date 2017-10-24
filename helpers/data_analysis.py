import numpy as np


def delete_bad_columns(x):

    bad_columns = []
    not_that_bad_columns =[]
    for i in range(len(x[0])):

        x_nan = x[:, i][x[:, i] == -999]

        nan_ratio = len(x_nan) / len(x[:, i])

        if nan_ratio > 0.6:
            bad_columns.append(i)

        elif nan_ratio > 0:
            not_that_bad_columns.append(i)

    tx = np.delete(x, bad_columns, 1)

    print("Bad Columns")
    print(bad_columns)
    print("\n\nNot That Bad Columns")
    print(not_that_bad_columns)

    return tx


def delete_bad_rows(x):

    bad_rows = []
    for i in range(len(x)):

        if -999 in x[i]:
            bad_rows.append(i)

    return np.delete(x, bad_rows, 0)


def replace_wrong_data(x):

    new_x = x

    tx = delete_bad_rows(x)

    for i in range(len(new_x[0])):

        mean = np.mean(tx[i])

        bad_indices = []
        for j in range(len(new_x)):
            if new_x[j, i] == -999:
                bad_indices.append(j)

        # print("\n\nBad Indices")
        # print(len(bad_indices))

        np.put(new_x[:, i], bad_indices, mean)

    return new_x


def features_normalization(x):

    new_x = x

    for i in range(len(new_x[0])):
        new_x[:, i] = (new_x[:, i] - np.mean(new_x[:, i])) / (np.max(new_x[:, i]) - np.min(new_x[:, i]))

    return new_x


