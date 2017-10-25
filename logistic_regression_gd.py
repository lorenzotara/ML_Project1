from helpers.data_analysis import *
from helpers.proj1_functions import *
from helpers.proj1_helpers import *

train_path = "Data/train.csv"

y, x, ids = load_csv_data(train_path)

######## DATA ANALYSIS ###########


x = replace_wrong_data(x)
x = features_normalization(x)

x_train, y_train, x_test, y_test = split_data(x, y, 0.7)


######## LOGISTIC REGRESSION ###########


initial_w = [0] * len(x[0])
max_iters = 1000

gamma = 0.0000001

losses, ws = logistic_regression2(y_train, x_train, initial_w, max_iters, gamma)

y_pred = predict_labels(ws, x_test)

final_result = y_test == y_pred

score = np.count_nonzero(final_result) / len(final_result)

print(score * 100, "%")
