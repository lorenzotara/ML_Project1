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

distribution_histogram(x_0)

'''We drop the last column because it's full of zeros'''
# x_0 = np.delete(x_0, x_0.shape[1] - 1, axis=1)
y_0 = y[np.where(x[:, jet_num] == 0)]
ids_0 = ids[np.where(x[:, jet_num] == 0)]


# x_0 = np.delete(x_0, jet_num, axis=1)

x_0 = delete_bad_columns(x_0)
x_0 = replace_wrong_data_mean(x_0)
x_0 = combine_features(x_0, np.arange(13))
x_0 = build_poly(x_0, 3)
x_0 = add_column_of_ones(x_0)
x_0[:, 1:len(x_0)] = features_standardization(x_0[:, 1:len(x_0)])

x_train, y_train, x_test, y_test = split_data(x_0, y_0, 0.7)
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

    cross_tr, cross_te = lr_ridge_cross_val_demo_lambda_fixed(y_0, x_0, 4, lambda_ / 100)
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
