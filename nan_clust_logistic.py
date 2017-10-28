import time

from helpers.data_analysis import *
from helpers.proj1_functions import *
from helpers.proj1_helpers import *
from matplotlib import pyplot as plt
from helpers.logistic_helpers import *

train_path = "data/train.csv"
test_path = "data/test.csv"

y, x, ids = load_csv_data(train_path)

y_test, x_test, ids_test = load_csv_data(test_path)

'''
We divide the dataset in two different clusters: 
one that contains NaNs and the other that not contains NaNs 
based on the feature DER_mass_MMC
'''
x_nan = x[np.where(x[:, 0] == -999)]
x_not_nan = x[np.where(x[:, 0] != -999)]
x_test_nan = x_test[np.where(x_test[:, 0] == -999)]
x_test_not_nan = x_test[np.where(x_test[:, 0] != -999)]

y_nan = y[np.where(x[:, 0] == -999)]
y_not_nan = y[np.where(x[:, 0] != -999)]
y_test_nan = y_test[np.where(x_test[:, 0] == -999)]
y_test_not_nan = y_test[np.where(x_test[:, 0] != -999)]

ids_nan = ids[np.where(x[:, 0] == -999)]
ids_not_nan = ids[np.where(x[:, 0] != -999)]
ids_test_nan = ids_test[np.where(x_test[:, 0] == -999)]
ids_test_not_nan = ids_test[np.where(x_test[:, 0] != -999)]

jet_num = 22

subsets_x = []
subsets_y = []
subsets_ids = []
subsets_x_test = []
subsets_y_test = []
subsets_ids_test = []
for i in range(4):
    '''
    We divide the dataset in two different clusters: 
    one that contains NaNs and the other that not contains NaNs 
    based on the feature DER_mass_MMC
    '''

    subsets_x.append(x_nan[np.where(x_nan[:, jet_num] == i)])
    subsets_x.append(x_not_nan[np.where(x_not_nan[:, jet_num] == i)])
    subsets_y.append(y_nan[np.where(x_nan[:, jet_num] == i)])
    subsets_y.append(y_not_nan[np.where(x_not_nan[:, jet_num] == i)])
    subsets_ids.append(ids_nan[np.where(x_nan[:, jet_num] == i)])
    subsets_ids.append(ids_not_nan[np.where(x_not_nan[:, jet_num] == i)])

    subsets_x_test.append(x_test_nan[np.where(x_test_nan[:, jet_num] == i)])
    subsets_x_test.append(x_test_not_nan[np.where(x_test_not_nan[:, jet_num] == i)])
    subsets_y_test.append(y_test_nan[np.where(x_test_nan[:, jet_num] == i)])
    subsets_y_test.append(y_test_not_nan[np.where(x_test_not_nan[:, jet_num] == i)])
    subsets_ids_test.append(ids_test_nan[np.where(x_test_nan[:, jet_num] == i)])
    subsets_ids_test.append(ids_test_not_nan[np.where(x_test_not_nan[:, jet_num] == i)])

predictions = []
final_ids = []

start = time.time()
subset_index = 0

lambdas = np.array([0.6, 0.9, 0.5, 0.1, 0.3, 0.4, 0.1, 0.1])


def test_gamma(x_set_train, y_set_train, ids_set, x_set_test, y_set_test, ids_set_test, lambda_subs, gamma):

    '''
    Working on the train data
    '''
    # PER LOGISTIC METTO ZERO
    y_set_train[np.argwhere(y_set_train == -1)] = 0

    x_set_train = np.delete(x_set_train, jet_num, axis=1)

    x_set_train = delete_bad_columns(x_set_train)
    x_set_train = delete_equal_columns(x_set_train)
    x_set_train = replace_wrong_data(x_set_train)
    # x_combined = combine_features(x_set, np.arange(13))
    # x_set = features_standardization(x_set)
    pca, eig_ratios = PCA(features_standardization(x_set_train), 14)

    x_sin = np.sin(x_set_train)
    x_cos = np.cos(x_set_train)
    x_set_train = np.c_[x_set_train, x_sin]
    x_set_train = np.c_[x_set_train, x_cos]

    x_set_train = build_poly(x_set_train, 4)

    x_set_train = add_column_of_ones(x_set_train)

    x_set_train[:, 1:len(x_set_train)] = features_standardization(x_set_train[:, 1:len(x_set_train)])

    x_set_train = np.c_[x_set_train, pca]
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

    print("x_set_test shape")
    print(x_set_test.shape)


    initial_w = np.zeros(x_set_train.shape[1])

    losses, ws = learning_with_penalized_logistic_gradient_desc(y_set_train, x_set_train, initial_w, gamma, lambda_subs,
                                                                2000)  # penalized

    loss_tr = (calculate_logistic_penalized_loss(y_set_train, x_set_train, ws, lambda_subs))
    # loss_te = (calculate_logistic_penalized_loss(y_test, x_test, ws, lambda_subs))

    print("loss_tr: ", loss_tr)


#test_gamma(subsets_x[7], subsets_y[7], subsets_ids[7], subsets_x_test[7], subsets_y_test[7], subsets_ids_test[7], lambdas[7], 0.6)

gammas = np.array([1.1, 0.8, 1, 0.4, 1, 0.6, 1.6, 0.6])


for x_set_train, y_set_train, ids_set, x_set_test, y_set_test, ids_set_test, lambda_subs, gamma in zip(subsets_x, subsets_y, subsets_ids,
                                                                                                subsets_x_test, subsets_y_test,
                                                                                                subsets_ids_test, lambdas, gammas):
    subset_index += 1
    
    # PER LOGISTIC METTO ZERO
    y_set_train[np.argwhere(y_set_train == -1)] = 0

    x_set_train = np.delete(x_set_train, jet_num, axis=1)

    x_set_train = delete_bad_columns(x_set_train)
    x_set_train = delete_equal_columns(x_set_train)
    x_set_train = replace_wrong_data(x_set_train)
    # x_combined = combine_features(x_set, np.arange(13))
    # x_set = features_standardization(x_set)
    pca, eig_ratios = PCA(features_standardization(x_set_train), 14)

    x_sin = np.sin(x_set_train)
    x_cos = np.cos(x_set_train)
    x_set_train = np.c_[x_set_train, x_sin]
    x_set_train = np.c_[x_set_train, x_cos]

    x_set_train = build_poly(x_set_train, 4)

    x_set_train = add_column_of_ones(x_set_train)

    x_set_train[:, 1:len(x_set_train)] = features_standardization(x_set_train[:, 1:len(x_set_train)])

    x_set_train = np.c_[x_set_train, pca]
    # x_set = np.c_[x_set, x_combined]

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

    #rmse_tr = []
    #rmse_te = []

    #cross_tr, cross_te, weight = cross_validation_penalized_logistic(y_set, x_set, np.zeros(x_set.shape[1]), 0.5, 4, lambda_subs, 500, False)

    #rmse_te.append(np.sqrt(cross_te))
    #rmse_tr.append(np.sqrt(cross_tr))
    

    #index = np.argmin(rmse_te)
    initial_w = np.zeros(x_set_train.shape[1])
    losses, ws = learning_with_penalized_logistic_gradient_desc(y_set_train, x_set_train, initial_w, gamma, lambda_subs, 2000) #penalized

    loss_tr = (calculate_logistic_penalized_loss(y_set_train, x_set_train, ws, lambda_subs))
    #loss_te = (calculate_logistic_penalized_loss(y_test, x_test, ws, lambda_subs))

    print("loss_tr: ", loss_tr)
    #print("loss_te: ", loss_te)

    y_prediction = predict_labels_for_lr(ws, x_set_test)
    predictions.append(y_prediction)
    final_ids.append(ids_set_test)

end = time.time()

print((end - start))

y_pred = np.concatenate(predictions)
indices = np.concatenate(final_ids)

# print(y_pred.shape)
# print(indices.shape)

create_csv_submission(indices, y_pred, "nan_clustering_logistic")
'''






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
