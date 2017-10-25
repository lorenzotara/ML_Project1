from helpers.data_analysis import *
from helpers.proj1_functions import *
from helpers.proj1_helpers import *

train_path = "Data/train.csv"

y, x, ids = load_csv_data(train_path)


######## DATA ANALYSIS ###########

# x = delete_bad_columns(x)

'''
If we delete the rows that contain -999 we have a worse score than if we didn't drop them
'''
# x = delete_bad_rows(x)
# y = delete_bad_rows(y)


'''
For the moment we replace wrong data with approximate mean of the column,
calculated without taking account of wrong data

In the future we shouldn't delete the row with wrong data, but we will have to
calculate the mean of every column without the wrong data. Because now we don't have elements in most of the columns
that were dropped because they were part of a row with a wrong element.
'''

x = replace_wrong_data(x)
prova = "link to git"

'''
Removes outliers
'''

x = outliers_modified_z_score(x)

'''
Features normalization
'''

x = features_normalization(x)


########################################

########## SPLIT DATA ##################

x_train, y_train, x_test, y_test = split_data(x, y, 0.7)


########################################


######## LINEAR REGRESSION ###########


initial_w = [0] * len(x[0])
max_iters = 1000

# gamma = 1e-6 * 1.5 it is the best when we drop columns and not rows (70.13% of score)

# gamma = 1e-5 it is good when we drop columns and replace the wrong elements with the mean

# gamma = 3 it is good when we normalize but normalization does not seem to work very well

# gamma = 1.65 when we don't delete anything and we put the mean to fill wrong values and we normalize: 71.18 %
gamma = 1.65

losses, ws = linear_regression(y_train, x_train, initial_w, max_iters, gamma)

y_pred = predict_labels(ws, x_test)

final_result = y_test == y_pred

print(np.count_nonzero(final_result)/len(final_result) * 100 + "%")

########################################






