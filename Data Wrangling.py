import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

URL = "/Users/lorenzotara/Documents/EPFL/Machine Learning/ML_course/projects/project1/Data/train.csv"

raw_x = pd.read_csv(URL, na_values=[-999])


y = raw_x.Prediction

der_x = raw_x.filter(regex="DER.*")

print(der_x.shape)

dropped_columns = []

'''
If we drop columns which values are nan for more than 50%
we lose 4 columns:
['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_lep_eta_centrality']
If we drop columns which values are more than 0.152 we also lose DER_mass_MMC

Dropping columns and rows:
70%:
['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_lep_eta_centrality', 'DER_deltaeta_jet_jet',
 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_lep_eta_centrality', 'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta',
  'PRI_jet_subleading_phi']


'''
for column in der_x.columns.values:

    if der_x[column].isnull().sum() > der_x.shape[0] * 0.70:
        der_x = der_x.drop(column, axis=1)
        dropped_columns.append(column)



for column in raw_x.columns.values:

    if raw_x[column].isnull().sum() > raw_x.shape[0] * 0.70:
        raw_x = raw_x.drop(column, axis=1)
        dropped_columns.append(column)

print(raw_x.shape)
print(dropped_columns)
print(raw_x.isnull().sum())
print(raw_x.dropna().shape)

der_x = der_x.fillna(0)
der_x = der_x.as_matrix()
y = y.as_matrix()



logistic = LogisticRegression()
logistic.fit(der_x[: int(len(der_x)*0.7)],y[:int(len(y)*0.7)])
Y = logistic.predict(der_x[int(len(der_x)*0.7):])

result = []
for i in range(len(Y)):
    result.append(Y[i] == y[int(len(y)*0.7) + i])


print(result.count(True)/len(result) * 100)


