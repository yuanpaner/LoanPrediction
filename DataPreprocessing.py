# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def read_data(path):
    """read npy file function"""
    data = np.load(path) # ndarray
    column_name = data[0].decode('UTF-8').strip().split(',')
    data_list = []
    index_list = []
    for train in data[1:]:
        tmp = train.decode('UTF-8').split(',')
        index = int(tmp[0])
        tmp = tmp[1:]
        tmp = [np.nan if t == 'NA' else float(t) for t in tmp]
        data_list.append(tmp)
        index_list.append(index)
    return index_list, column_name, data_list

# Importing the dataset
filename = 'ecs171train.npy'

# Read in data
if os.path.exists('data.df'):
    df = pd.read_pickle('data.df')
else:
    idx, col, data = read_data(filename)
    df = pd.DataFrame(data = data, columns = col[1:], index = idx)
    df.to_pickle('data.df')

#remove columns with same values in all rows
nunique = df.apply(pd.Series.nunique) # pandas.core.series.Series
drop_cols = nunique[nunique == 1].index # ['f33', 'f34', 'f35', 'f37', 'f38', 'f678', 'f700', 'f701', 'f702', 'f736', 'f764']
df.drop(drop_cols,axis=1,inplace=True) # feature # 770 -> 759




# Missing Data

# Scalar


# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set

# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
