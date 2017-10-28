from helpers.data_analysis import *
from helpers.proj1_functions import *
from helpers.proj1_helpers import *
from matplotlib import pyplot as plt

train_path = "Data/train.csv"
test_path = "Data/test.csv"

y, x, ids = load_csv_data(train_path)

# x = np.c_[ids, x]
# y = np.c_[ids, y]


'''
'PRI_jet_num'
'''
jet_num = 22
x_list = []
y_list = []

# print(x[np.argwhere(x[:, 23] == 0)].shape)

x_0 = x[np.where(x[:, jet_num] == 0)]
# np.savetxt("x0.csv", x_0, delimiter=",")

'''We drop the last column because it's full of zeros'''
x_0 = np.delete(x_0, x_0.shape[1] - 1, axis=1)
y_0 = y[np.where(x[:, jet_num] == 0)]
ids_0 = ids[np.where(x[:, jet_num] == 0)]

x_list.append(x_0)
y_list.append(y_0)

x_1 = x[np.where(x[:, jet_num] == 1)]
y_1 = y[np.where(x[:, jet_num] == 1)]
ids_1 = ids[np.where(x[:, jet_num] == 1)]
# np.savetxt("x1.csv", x_1, delimiter=",")

x_list.append(x_1)
y_list.append(y_1)
# print(len(x_1))

x_2 = x[np.where(x[:, jet_num] == 2)]
y_2 = y[np.where(x[:, jet_num] == 2)]
ids_2 = ids[np.where(x[:, jet_num] == 2)]
# np.savetxt("x2.csv", x_2, delimiter=",")

x_list.append(x_2)
y_list.append(y_2)
# print(len(x_2))

x_3 = x[np.where(x[:, jet_num] == 3)]
y_3 = y[np.where(x[:, jet_num] == 3)]
ids_3 = ids[np.where(x[:, jet_num] == 3)]
# np.savetxt("x3.csv", x_3, delimiter=",")

x_list.append(x_3)
y_list.append(y_3)
# print(len(x_3))

# print(len(x_0) + len(x_1) + len(x_2) + len(x_3))


'''
Uncomment when you want to submit 
'''
# y_test, x_test, ids_test = load_csv_data(test_path)
#
# x_test = replace_wrong_data(x_test)
# x_test_sin = np.sin(x_test)
# x_test_cos = np.cos(x_test)
# x_test = np.c_[x_test, x_test_cos]
# x_test = np.c_[x_test, x_test_sin]
# x_test = build_poly(x_test, 4)
# x_test = add_column_of_ones(x_test)
# x_test[:, 1:len(x)] = features_standardization(x_test[:, 1:len(x)])



######## DATA ANALYSIS ###########

# x = delete_bad_columns(x)

'''
If we delete the rows that contain -999 we have a worse score than if we didn't drop them
'''
# print(x)
# x, y = delete_bad_rows(x, y)



'''
For the moment we replace wrong data with approximate mean of the column,
calculated without taking account of wrong data

'''

# x = replace_wrong_data(x)
#
# x_sin = np.sin(x)
# x_cos = np.cos(x)
# # x_exp = np.exp(x)
#
# # x = combine_features(x, np.arange(4))
# x = build_poly(x, 4)
# x = np.c_[x, x_cos]
# x = np.c_[x, x_sin]
# x = add_column_of_ones(x)



# np.savetxt("x0.csv", x_0, delimiter=",")

predictions = []


for x_elem, y_elem in zip(x_list, y_list):

    x_elem = np.delete(x_elem, jet_num, axis=1)

    x_elem = delete_bad_columns(x_elem)
    x_elem = replace_wrong_data_mean(x_elem)
    x_elem = combine_features(x_elem, np.arange(13))
    x_elem = build_poly(x_elem, 3)
    x_elem = add_column_of_ones(x_elem)
    x_elem[:, 1:len(x_elem)] = features_standardization(x_elem[:, 1:len(x_elem)])

    x_train, y_train, x_test, y_test = split_data(x_elem, y_elem, 0.7)

    # lambda_ = 0.8
    '''
BEST LAMBDA:  240
84.35644224994996 %
BEST LAMBDA:  240
79.99914030261348 %
BEST LAMBDA:  240
82.8172555246791 %
BEST LAMBDA:  240
82.796992481203 %
    '''
    # Without cross validation
    # losses = []
    # weights = []
    # lambdas = []
    # for lambda_ in range(0, 200, 10):
    #     loss, ws = ridge_regression(y_train, x_train, lambda_/100)
    #     losses.append(loss)
    #     weights.append(ws)
    #     lambdas.append(lambda_)
    #
    # index = np.argmin(losses)
    # loss = losses[index]
    # print("BEST LAMBDA: ", lambdas[index])
    # ws = weights[index]

    '''
    CROSS VALIDATION
    x_0: lambda = 1.85
    '''

    lambdas = []
    rmse_tr = []
    rmse_te = []
    for lambda_ in range(0, 200, 10):

        cross_tr, cross_te = lr_ridge_cross_val_demo_lambda_fixed(y_elem, x_elem, 4, lambda_/100)
        lambdas.append(lambda_/100)
        rmse_tr.append(np.mean(cross_tr))
        rmse_te.append(np.mean(cross_te))

    plt.figure()
    ax = plt.subplot(111)
    training_plot = ax.plot(lambdas, rmse_tr)
    testing_plot = ax.plot(lambdas, rmse_te)
    ax.set_xlabel("lambdas")
    ax.set_ylabel("errors")
    ax.legend((training_plot[0], testing_plot[0]), ("training error", "testing error"))
    plt.show()

    # y_pred = predict_labels(ws, x_test)
    #
    # final_result = y_test == y_pred
    # score = np.count_nonzero(final_result) / len(final_result)
    # print(score * 100, "%")
    # predictions.append(y_pred)






# final_result = y_test == y_pred
#
# score = np.count_nonzero(final_result) / len(final_result)
#
# print(score * 100, "%")