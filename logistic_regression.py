from helpers.data_analysis import *
from helpers.proj1_functions import *
from helpers.proj1_helpers import *

train_path = "Data/train.csv"

y, x, ids = load_csv_data(train_path)

######## DATA ANALYSIS ###########


x = replace_wrong_data(x)
x = features_standardization(x)

x_train, y_train, x_test, y_test = split_data(x, y, 0.7)


######## LOGISTIC REGRESSION ###########


initial_w = [0] * len(x[0])
max_iters = 1000

# gamma = 1e-6 * 1.5 it is the best when we drop columns and not rows (70.13% of score)

# gamma = 1e-5 it is good when we drop columns and replace the wrong elements with the mean

# gamma = 3 it is good when we normalize but normalization does not seem to work very well

# gamma = 1.65 when we don't delete anything and we put the mean to fill wrong values and we normalize: 71.18 %
gamma = 1.65

losses, ws = logistic_regression2(y_train, x_train, initial_w, max_iters, gamma)

y_pred = predict_labels(ws, x_test)

final_result = y_test == y_pred

print(np.count_nonzero(final_result)/len(final_result) * 100 + "%")
