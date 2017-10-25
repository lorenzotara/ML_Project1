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


def outliers_modified_z_score(xs):
    """
    finds the index of the outliers, and replaces them by the mean value of the column.
    :type xs: feature matrix
    """
    for i in range(len(xs[0])):
        threshold = 3.5

        median_x = np.median(xs[:, i])
        # modified_z_scores = []

        for x in xs[:, i]:
            median_absolute_deviation_x = np.median(np.abs(x - median_x))
            if median_absolute_deviation_x == 0:
                median_absolute_deviation_x = 1e-30
            modified_z_scores = (0.6745 * (x - median_x) / median_absolute_deviation_x)

            if modified_z_scores > threshold:
                x[:, i] = mean(xs[:, i]) # * random number(either based on distribution or Q1/Q3) - avoid overemphasizing mean
        outliers = np.array(np.where(np.abs(modified_z_scores) > threshold))

        # median_x = np.median(xs[:, i])
        # median_absolute_deviation_x = np.median([np.abs(x - median_x) for x in xs[:, i]])
        # # NEED TO FIX: DIVISION BY ZERO:
        # modified_z_scores = []
        # for x in xs[:, i]:
        #     if median_absolute_deviation_x == 0:
        #         median_absolute_deviation_x = 1e-10
        #     modified_z_scores.append(0.6745 * (x - median_x) / median_absolute_deviation_x)
        # outliers = np.array(np.where(np.abs(modified_z_scores) > threshold))
        #
        # outliers = np.reshape(outliers, [len(outliers[0, :]), ])
        # temp = np.delete(xs[:, i], outliers)
        #
        # for j in enumerate(outliers):
        #     xs[j, i] = np.mean(temp)
    return xs

def distribution_histogram(x):
    for i in x[:, i]:
        plt.figure(i)
        plt.plot()