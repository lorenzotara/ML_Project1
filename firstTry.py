from proj1_helpers import *
from proj1_functions import *

import numpy as np

train_path = "Data/train.csv"

y, x, ids = load_csv_data(train_path)


######## DATA ANALYSIS ###########



bad_columns = []
not_that_bad_columns =[]
for i in range(len(x[0])):

    x_nan = x[:, i][x[:, i] == -999]

    nan_ratio = len(x_nan) / len(x[:, i])

    if nan_ratio > 0.6:
        bad_columns.append(i)

    elif nan_ratio > 0:
        not_that_bad_columns.append(i)

x = np.delete(x, bad_columns, 1)

print("Bad Columns")
print(bad_columns)
print("\n\nNot That Bad Columns")
print(not_that_bad_columns)


bad_rows = []
for i in range(len(x)):

    if -999 in x[i]:
        bad_rows.append(i)

'''
If we delete the rows that contain -999 we have a worse score than if we didn't drop them
'''

# x = np.delete(x, bad_rows, 0)
# y = np.delete(y, bad_rows, 0)
# print(len(x))


'''
For the moment we replace wrong data with approximate mean of the column,
calculated without taking account of wrong data

In the future we shouldn't delete the row with wrong data, but we will have to
calculate the mean of every column without the wrong data. Because now we don't have elements in most of the columns
that were dropped because they were part of a row with a wrong element.
'''

tx = np.delete(x, bad_rows, 0)

for i in range(len(x[0])):

    mean = np.mean(tx[i])

    bad_indices = []
    for j in range(len(x)):
        if x[j, i] == -999:
            bad_indices.append(j)

    # print("\n\nBad Indices")
    # print(len(bad_indices))

    np.put(x[:, i], bad_indices, mean)


'''
Features normalization
'''

for i in range(len(x[0])):

    x[:, i] = (x[:, i] - np.mean(x[:, i])) / (np.max(x[:, i]) - np.min(x[:, i]))


########################################

########## SPLIT DATA ##################

x_train, y_train, x_test, y_test = split_data(x, y, 0.7)


########################################


######## GRADIENT DESCENT ###########


initial_w = [0] * len(x[0])
max_iters = 1000
# gamma = 1e-6 * 1.5 it is the best when we drop columns and not rows (70.13% of score)
# gamma = 1e-5 it is good when we drop columns and replace the wrong elements with the mean
# gamma = 3 it is good when we normalize but normalization does not seem to work very well
gamma = 3

losses, ws = gradient_descent(y_train, x_train, initial_w, max_iters, gamma)

y_pred = predict_labels(ws, x_test)

final_result = y_test == y_pred

print(np.count_nonzero(final_result)/len(final_result) * 100)

########################################






