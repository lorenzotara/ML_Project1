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


