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

if os.path.exists('data.df'):
    df = pd.read_pickle('data.df')
else:
    idx, col, data = read_data(filename)
    df = pd.DataFrame(data = data, columns = col[1:], index = idx)
    df.to_pickle('data.df')




#remove columns with same values in all rows
nunique = df.apply(pd.Series.nunique) # pandas.core.series.Series
drop_cols = nunique[nunique == 1].index # ['f33', 'f34', 'f35', 'f37', 'f38', 'f678', 'f700', 'f701', 'f702', 'f736', 'f764']
df.drop(drop_cols,axis=1,inplace=True) # feature+y # 770 -> 759





# Encoding categorical data to avoid one answer is greater/more weighted than others when represented by numbers, no need here
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# onehotencoder = OneHotEncoder(categorical_features = [0])
# X = onehotencoder.fit_transform(X).toarray()
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)
categorical_cols = ['f776', 'f777']






X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# Missing Data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=np.nan , strategy='mean', axis=0)
X = imp.fit_transform(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
""" many ML models are based on Euclidean Distance
    which algorithms : https://stats.stackexchange.com/questions/244507/what-algorithms-need-feature-scaling-beside-from-svm
    1. scale 'fit' on and transform to training set, and transform to test set.
    2. do we need to scale dummy vars? depends. interpretation
"""

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
