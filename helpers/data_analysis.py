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

    tx = []
    # Every element of tx is a column of x without the wrong data
    for i in range(len(x[0])):

        tx.append(np.delete(x[:, i], np.argwhere(x[:, i] == -999)))

    # Calculating the mean of every column not taking account of the wrong data
    # and then putting it instead of the wrong datum
    for i in range(len(new_x[0])):

        mean = np.mean(tx[i])

        new_x[np.argwhere(new_x[:, i] == -999), i] = mean

    return new_x


def features_standardization(x):

    new_x = x

    for i in range(len(new_x[0])):

        # Dividing by the standard deviation
        new_x[:, i] = (new_x[:, i] - np.mean(new_x[:, i])) / np.std(new_x[:, i])

    return new_x


def features_normalization(x):

    new_x = x

    for i in range(len(new_x[0])):
        # Dividing by the difference between the maximum and the minimum
        new_x[:, i] = (new_x[:, i] - np.mean(new_x[:, i])) / (np.max(new_x[:, i]) - np.min(new_x[:, i]))

    return new_x


