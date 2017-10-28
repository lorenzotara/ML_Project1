import numpy as np
from matplotlib import pyplot as plt


def delete_bad_columns(x):
    '''Deleting every column that have more than 60% of wrong values'''

    bad_columns = []
    not_that_bad_columns =[]
    for i in range(len(x[0])):

        x_nan = x[:, i][x[:, i] == -999]

        nan_ratio = len(x_nan) / len(x[:, i])

        if nan_ratio > 0.99:
            bad_columns.append(i)

        elif nan_ratio > 0:
            not_that_bad_columns.append(i)

    tx = np.delete(x, bad_columns, 1)

    # print("Bad Columns")
    # print(bad_columns)
    # print("\n\nNot That Bad Columns")
    # print(not_that_bad_columns)

    return tx


def delete_bad_rows(x, y):
    '''Deleting every row with wrong values'''

    bad_rows = []

    for i in range(len(x)):

        if -999 in x[i]:
            bad_rows.append(i)

    return np.delete(x, bad_rows, 0), np.delete(y, bad_rows)


def delete_equal_columns(x):

    temp = x
    columns_to_del = []
    for index in range(x.shape[1]):
        if np.std(x[:, index]) == 0:
            # temp = np.delete(temp, index, axis=1)
            columns_to_del.append(index)

    return np.delete(temp, columns_to_del, axis=1)



def replace_wrong_data(x):
    '''Replacing every wrong value with the mean of the column calculated without those values'''
    new_x = x

    tx = []
    # Every element of tx is a column of x without the wrong data
    for i in range(len(x[0])):

        tx.append(np.delete(x[:, i], np.where(x[:, i] == -999)))

    # Calculating the mean of every column not taking account of the wrong data
    # and then putting it instead of the wrong datum
    for i in range(len(new_x[0])):

        mean = np.mean(tx[i])

        new_x[np.where(new_x[:, i] == -999), i] = mean

    return new_x


def combine_features(x, list_of_features):
    '''Combining every feature in list_of_features'''

    features_combined = []
    new_x = x

    for column1 in list_of_features:
        for column2 in list_of_features:

            '''If the feature has not been seen already, we combine it with all the others'''
            if (column1 not in features_combined) & (column2 not in features_combined) & (column1 != column2):

                new_x = np.c_[new_x, new_x[:, column1] * new_x[:, column2]]

        features_combined.append(column1)

    return new_x

def calculate_mean_std_vector(x):
    mu = []
    sigma = []

    for i in range(len(x[0])):

        # Creates the mean and standard deviation vectors
        mu.append(np.mean(x[:, i]))
        sigma.append(np.std(x[:, i]))

    return mu, sigma

def features_standardization(x):
    '''Standardizing the features'''
    new_x = x

    for i in range(len(new_x[0])):

        # Subtracting the mean and dividing by the standard deviation for every column
        new_x[:, i] = (new_x[:, i] - np.mean(new_x[:, i])) / np.std(new_x[:, i])

    return new_x

def rescale_standardization(x, mu, sigma):
    '''Scales input back to original form, based on original mean and standard deviation.'''
    new_x = x

    for i in range(len(new_x[0])):

        # Adding the mean and multiply by the standard deviation for every column
        new_x[:, i] = (new_x[:, i] + mu) * sigma

    return new_x

def features_normalization(x):
    '''Normalizing the features'''

    new_x = x

    for i in range(len(new_x[0])):
        # Subtracting the mean and dividing by the difference between the maximum and the minimum
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

        tx.append(np.delete(x[:, i], np.where(x[:, i] == -999)))

        plt.figure(i)
        plt.hist(tx[i], bins=150)  # arguments are passed to np.histogram
        plt.title("{y}".format(y=title))
        plt.show()


def PCA(x, chosen_dimensions):
    ''' Computes the principal components of a given data set. Input x has to be without NaNs and standardized,
    chosen_dimensions is the dimensions one wishes to end up with. Returns the projected X with the chosen dimensions,
    and the eigenvalue ratios for plotting.
    '''
    # W: vector with all the eigenvalues, V: array of eigenvectors
    W, V = np.linalg.eig(np.cov(x, rowvar=False))  # rowvar=False: column - variable, rows - observations.

    #Analysing the eigenvalues
    W_sort = np.sort(W)
    W_sort = list(W_sort[::-1])  # flip w in descending order
    eig_ratios = W_sort / sum(W)  # ratios of eigenvalues


    #Sorts the eigenvectors corresponding to sorted eigenvalues
    W = list(W)
    PC_indices = []
    V_sort = np.zeros([len(V)])
    for i in range(len(W_sort)):
        # PC_indices.append(W.index(w))
        PC_indices.append(W.index(W_sort[i]))
        V_sort = np.column_stack([V_sort, V[:, PC_indices[i]]])
    V_sort = np.delete(V_sort, [0], axis=1)

    projected_data = np.dot(x, V_sort[:, :chosen_dimensions])

    return projected_data, eig_ratios


def visualization_PCA(eig_ratios):
    '''Plots the eigenvalue ratios of the principal components and the cumulated ratios in a scree plot'''
    # X-axis labels (xticks)
    objects = ('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13',
               'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20', 'PC21', 'PC22', 'PC23', 'PC24', 'PC25',
               'PC26', 'PC27', 'PC28', 'PC29', 'PC30')
    N = len(objects)

    #Length of x-axis
    x_pos = np.arange(N)

    #Computes accumulated eigenvector ratios
    eig_accumulated = np.cumsum(eig_ratios, dtype=float)

    #Scree plot
    plt.figure(1)
    plt.scatter(x_pos, eig_ratios)
    plt.scatter(x_pos, eig_accumulated)
    plt.plot(x_pos, eig_ratios)
    plt.plot(x_pos, eig_accumulated)
    plt.axhline(y=0.85, color='r')
    plt.axvline(x=13, color='r')
    plt.grid()
    plt.ylabel('Compared value')
    plt.title('Eigenvalues, PCA')

    # Following fits xticks, so all xticks are readable.
    plt.xticks(x_pos, objects)

    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2  # inch margin
    s = maxsize / plt.gcf().dpi * (N + 1) + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    plt.show()


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def linear_interpolation(x):
    '''linear interpolation of NaNs'''

    nans, indices = nan_helper(x[:, 0])

    x[: 0][nans]= np.interp(indices(nans), indices(~nans), x[:, 0][~nans])
