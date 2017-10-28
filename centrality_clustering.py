from helpers.data_analysis import *
from helpers.proj1_functions import *
from helpers.proj1_helpers import *
from matplotlib import pyplot as plt

train_path = "Data/train.csv"
test_path = "Data/test.csv"

y, x, ids = load_csv_data(train_path)

y_test, x_test, ids_test = load_csv_data(test_path)



jet_num = 22

subsets_x = []
subsets_y = []
subsets_ids = []
subsets_x_test = []
subsets_y_test = []
subsets_ids_test = []
for i in range(2):
    '''
    Creating first two clusters based on PRI_jet_num 0 and 1, and dividing those two into four based on nan or non-nan
    values in the DER_mass_MMD column.
    '''

    subsets_x.append(x[np.where(x[:, jet_num] == i & x[:, 0] == -999)])
    subsets_x.append(x[np.where(x[:, jet_num] == i & x[:, 0] != -999)])
    subsets_y.append(y[np.where(x[:, jet_num] == i & x[:, 0] == -999)])
    subsets_y.append(y[np.where(x[:, jet_num] == i & x[:, 0] != -999)])
    subsets_ids.append(ids[np.where(x[:, jet_num] == i & x[:, 0] == -999)])
    subsets_ids.append(ids[np.where(x[:, jet_num] == i & x[:, 0] != -999)])

    subsets_x_test.append(x_test[np.where(x_test[:, jet_num] == i & x[:, 0] == -999)])
    subsets_x_test.append(x_test[np.where(x_test[:, jet_num] == i & x[:, 0] != -999)])
    subsets_y_test.append(y_test[np.where(x_test[:, jet_num] == i & x[:, 0] == -999)])
    subsets_y_test.append(y_test[np.where(x_test[:, jet_num] == i & x[:, 0] != -999)])
    subsets_ids_test.append(ids_test[np.where(x_test[:, jet_num] == i & x[:, 0] == -999)])
    subsets_ids_test.append(ids_test[np.where(x_test[:, jet_num] == i & x[:, 0] != -999)])

'''Split the jet numbers 2 and 3 based on the DER_lep_eta_centrality greater or smaller than 0.5'''

subsets_x.append(x[np.where(x[:, jet_num] > 1 & x[:, 12] > 0.5)])
subsets_x.append(x[np.where(x[:, jet_num] > 1 & x[:, 12] < 0.5)])
subsets_y.append(y[np.where(x[:, jet_num] > 1 & x[:, 12] > 0.5)])
subsets_y.append(y[np.where(x[:, jet_num] > 1 & x[:, 12] < 0.5)])
subsets_ids.append(ids[np.where(x[:, jet_num] > 1 & x[:, 12] > 0.5)])
subsets_ids.append(ids[np.where(x[:, jet_num] > 1 & x[:, 12] < 0.5)])

subsets_x_test.append(x_test[np.where(x_test[:, jet_num] > 1 & x_test[:, 12] > 0.5)])
subsets_x_test.append(x_test[np.where(x_test[:, jet_num] > 1 & x_test[:, 12] < 0.5)])
subsets_y_test.append(y_test[np.where(x_test[:, jet_num] > 1 & x_test[:, 12] > 0.5)])
subsets_y_test.append(y_test[np.where(x_test[:, jet_num] > 1 & x_test[:, 12] < 0.5)])
subsets_ids_test.append(ids_test[np.where(x_test[:, jet_num] > 1 & x_test[:, 12] > 0.5)])
subsets_ids_test.append(ids_test[np.where(x_test[:, jet_num] > 1 & x_test[:, 12] < 0.5)])


predictions = []
final_ids = []

for x_set, y_set, ids_set, x_set_test, y_set_test, ids_set_test in zip(subsets_x, subsets_y, subsets_ids, subsets_x_test, subsets_y_test, subsets_ids_test):

    '''
    Working on the train data
    '''

    x_set = np.delete(x_set, jet_num, axis=1)

    x_set = delete_bad_columns(x_set)
    x_set = delete_equal_columns(x_set)
    x_set = replace_wrong_data(x_set)
    # x_combined = combine_features(x_set, np.arange(13))
    # x_set = features_standardization(x_set)
    pca, eig_ratios = PCA(features_standardization(x_set), 14)

    x_sin = np.sin(x_set)
    x_cos = np.cos(x_set)
    x_set = np.c_[x_set, x_sin]
    x_set = np.c_[x_set, x_cos]

    x_set = build_poly(x_set, 4)

    x_set = add_column_of_ones(x_set)

    x_set[:, 1:len(x_set)] = features_standardization(x_set[:, 1:len(x_set)])

    x_set = np.c_[x_set, pca]
    # x_set = np.c_[x_set, x_combined]

    '''
    Working on the test data
    '''

    x_set_test = np.delete(x_set_test, jet_num, axis=1)

    x_set_test = delete_bad_columns(x_set_test)
    x_set_test = delete_equal_columns(x_set_test)
    x_set_test = replace_wrong_data(x_set_test)

    pca_test, eig_ratios_test = PCA(features_standardization(x_set_test), 14)

    x_sin_test = np.sin(x_set_test)
    x_cos_test = np.cos(x_set_test)
    x_set_test = np.c_[x_set_test, x_sin_test]
    x_set_test = np.c_[x_set_test, x_cos_test]

    x_set_test = build_poly(x_set_test, 4)

    x_set_test = add_column_of_ones(x_set_test)

    x_set_test[:, 1:len(x_set_test)] = features_standardization(x_set_test[:, 1:len(x_set_test)])

    x_set_test = np.c_[x_set_test, pca_test]


    '''
    CROSS VALIDATION
    x_0: lambda = 1.85
    '''

    lambdas = []
    rmse_tr = []
    rmse_te = []
    for lambda_ in range(0, 200, 10):

        cross_tr, cross_te = lr_ridge_cross_val_demo_lambda_fixed(y_set, x_set, 4, lambda_/100)
        lambdas.append(lambda_/100)
        rmse_tr.append(np.mean(cross_tr))
        rmse_te.append(np.mean(cross_te))

    '''
    Plotting the errors from cross validation
    '''
    plt.figure()
    ax = plt.subplot(111)
    training_plot = ax.plot(lambdas, rmse_tr)
    testing_plot = ax.plot(lambdas, rmse_te)
    ax.set_xlabel("lambdas")
    ax.set_ylabel("errors")
    ax.legend((training_plot[0], testing_plot[0]), ("training error", "testing error"))
    plt.show()
    plt.close()

    index = np.argmin(rmse_te)
    lambda_ = lambdas[index]

    losses, ws = ridge_regression(y_set, x_set, lambda_)

    # predictions.append(predict_labels(ws, x_set_test))
    # final_ids.append(ids_set_test)
    print('best lambda:', lambda_)
    # print('losses:', losses)




# y_pred = np.concatenate(predictions)
# indices = np.concatenate(final_ids)
#
# print(y_pred.shape)
# print(indices.shape)
#
# create_csv_submission(indices, y_pred, "nan_clustering")







# '''
# We divide the dataset in two different clusters:
# one that contains NaNs and the other that not contains NaNs
# based on the feature DER_mass_MMC
# '''
# x_nan = x[np.where(x[:, 0] == -999)]
# x_not_nan = x[np.where(x[:, 0] != -999)]
# y_nan = x[np.where(x[:, 0] == -999)]
# y_not_nan = x[np.where(x[:, 0] != -999)]
# ids_nan = x[np.where(x[:, 0] == -999)]
# ids_not_nan = x[np.where(x[:, 0] != -999)]
#
# subsets_x.append(x_nan)
# subsets_x.append(x_not_nan)
# subsets_y.append(y_nan)
# subsets_y.append(y_not_nan)
# subsets_ids.append(ids_nan)
# subsets_ids.append(ids_not_nan)
#
# jet_num = 22
#
# for set_index in range(len(subsets_x)):
#
#     for i in range(4):
#         '''
#         We divide every subset based on the PRI_jet_num feature value
#         '''
