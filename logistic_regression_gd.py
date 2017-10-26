from helpers.data_analysis import *
from helpers.proj1_functions import *
from helpers.proj1_helpers import *

train_path = "Data/train.csv"

y, x, ids = load_csv_data(train_path)
y[np.argwhere(y == -1)] = 0
# print(y)

######## DATA ANALYSIS ###########


x = replace_wrong_data(x)
x = features_standardization(x)
# x = features_normalization(x)

x_train, y_train, x_test, y_test = split_data(x, y, 0.7)


######## LOGISTIC REGRESSION ###########


initial_w = [1.56706398e-02,  -8.55227232e-01,  -5.83699794e-02,   3.07598467e-01,
   3.83684552e-01,   4.10729646e-01,  -3.18501137e-01,   2.54621534e-01,
  -1.64218368e-01,   1.77109302e-01,  -4.39028493e-01,   5.23003851e-01,
   3.72901313e-01,   5.60193278e-01,  -8.89406780e-03,  -1.73239346e-02,
  -1.54590360e-02,   3.11195926e-03,   5.27885198e-03,  -4.37114241e-02,
   6.19846403e-03,   1.07777459e-01,   1.18997080e-01,   7.94781733e-02,
  -1.90523200e-03,   6.12725324e-03,  -1.69409397e-01,  -2.97653108e-04,
  -7.90327844e-03,   8.44624783e-02]

max_iters = 1000

gamma = 1

losses, ws = logistic_regression2(y_train, x_train, initial_w, max_iters, gamma)
print(ws)
y_pred = predict_labels_for_lr(ws, x_test)

final_result = y_test == y_pred

score = np.count_nonzero(final_result) / len(final_result)

print(score * 100, "%")
