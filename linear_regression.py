from matplotlib import pyplot as plt
from helpers.proj1_helpers import *
from helpers.data_analysis import *
from helpers.proj1_functions import *

train_path = "Data/train.csv"
test_path = "Data/test.csv"

y, x, ids = load_csv_data(train_path)

'''
Uncomment when you want to submit
'''
# y_test, x_test, ids_test = load_csv_data(test_path)
#
# x_test = replace_wrong_data(x_test)
# x_test_sin = np.sin(x_test)
# x_test_cos = np.cos(x_test)
# x_test = build_poly(x_test, 5)
# x_test = np.c_[x_test, x_test_cos]
# x_test = np.c_[x_test, x_test_sin]
# x_test = combine_features(x_test, np.arange(15))
# x_test = add_column_of_ones(x_test)

######## DATA ANALYSIS ###########

'''
When I delete the bad columns the gradient descent slows drastically down. WHY???????
'''
# x = delete_bad_columns(x)

'''
If we delete the rows that contain -999 we have a worse score than if we didn't drop them
'''
# x, y = delete_bad_rows(x, y)


'''
For the moment we replace wrong data with approximate mean of the column,
calculated without taking account of wrong data

'''

x = replace_wrong_data_mean(x)

x_sin = np.sin(x)
x_cos = np.cos(x)
# x_exp = np.exp(x)

x = build_poly(x, 5)
x = np.c_[x, x_cos]
x = np.c_[x, x_sin]
x = combine_features(x, np.arange(15))
x = add_column_of_ones(x)

'''
Features normalization/standardization - except first column of ones
'''

# x[:, 1:len(x)] = features_normalization(x[:, 1:len(x)])
x[:, 1:len(x)] = features_standardization(x[:, 1:len(x)])






########################################

########## SPLIT DATA ##################

x_train, y_train, x_test, y_test = split_data(x, y, 0.7)

'''
Uncomment when you want to submit
'''
# x_train = x
# y_train = y


########################################


######## LINEAR REGRESSION ###########


initial_w = [0] * len(x_train[0])
max_iters = 2000

# gamma = 0.3 replace_wrong_data and features_normalization 71.76 %
# gamma = 1.9 replace_wrong_data, build_poly(3) and features_normalization 74.672 % - 500 iter
# gamma = 1.9 replace_wrong_data, build_poly(3) and features_normalization 76.041 % - 2000 iter
# gamma = 0.13 replace_wrong_data, build_poly(3) and features_standardization 77.63 % - 500 iter
# gamma = 0.13 replace_wrong_data, build_poly(3) and features_standardization 78.18 % - 2000 iter !!!!!! BEST SCORE
'''
GRADIENT DESCENT
'''


'''
Plot the data with gamma as parameter.
'''

# plot_gamma_parameter(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, max_iters=max_iters,
#                      initial_w=initial_w, gamma_start=0, gamma_end=2, step_size=1)


gamma = 0.04

losses, ws = linear_regression(y_train, x_train, initial_w, max_iters, gamma)

y_pred = predict_labels(ws, x_test)


'''
Uncomment when you want to submit
'''
# create_csv_submission(ids_test, y_pred, "linear regression prediction")

final_result = y_test == y_pred

score = np.count_nonzero(final_result) / len(final_result)

print(score * 100, "%")

########################################






