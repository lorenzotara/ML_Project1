import numpy as np
from matplotlib import pyplot as plt


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


def delete_bad_rows(x, y):

    bad_rows = []

    for i in range(len(x)):

        if -999 in x[i]:
            bad_rows.append(i)

    return np.delete(x, bad_rows, 0), np.delete(y, bad_rows)


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
    titles = np.array(['DER_mass_MMC', 'DER_mass_transverse_met_lep',
                       'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
                       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
                       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'DER_lep_eta_centrality',
                       'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta',
                       'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_num',
                       'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi',
                       'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi',
                       'PRI_jet_all_pt'])


    tx = []
    for i, title in enumerate(titles):

        print(i)

        tx.append(np.delete(x[:, i], np.argwhere(x[:, i] == -999)))

        plt.figure(i)
        plt.hist(tx[i], bins=150)  # arguments are passed to np.histogram
        plt.title("{y}".format(y=title))
        plt.show()
