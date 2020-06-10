import pandas as pd
import numpy as np
import os

if not os.path.exists('data'):
    os.mkdir('data')

Xpd = pd.read_csv('data/days.csv', index_col = 'Date')
X = Xpd.values

valid_company_idx = np.where(np.sum(np.isnan(X), axis=0) == 10)[0]
company_names = Xpd.columns[valid_company_idx]
X = X[:,valid_company_idx]

valid_time_idx = np.where(np.isnan(X[:,0]) == False)[0]
X = X[valid_time_idx,:]

np.save('data/russel_data.npy', X)
np.save('data/company_names.npy', company_names)





