import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import cross_validation
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA

# import training data as data frame
#data = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/training_final_good.csv")
data = pd.read_csv("D:/Julian/64_ie_maps/cleaning_training/train_ff2.csv")

# replace -999 flags
data = data.replace(-999,np.nan)

data = data.loc\
		[\
		  (data.iloc[:,452] == 2004)\
		| (data.iloc[:,452] == 2005)\
		| (data.iloc[:,452] == 2006)\
		#| (data.iloc[:,452] == 2007)\
		| (data.iloc[:,452] == 2009)\
		| (data.iloc[:,452] == 2010)\
		| (data.iloc[:,452] == 2011)\
		| (data.iloc[:,452] == 2012)\
		| (data.iloc[:,452] == 2013)\
		]


# indices of variables of interest (target and covariates)
selectionx = np.array(range(4,420))
selectiony = np.array([427,428,433,438])
selection = np.append(selectiony,selectionx)
# select data of interest
data_selection = data.iloc[:,selection].as_matrix()

# check type of array
#print(np.dtype(data_selection))

# force dtype = float32
data_selection = data_selection.astype(np.float32, copy=False)

# complete cases
data_selection = data_selection[~np.isnan(data_selection).any(axis=1)]
data_selection = data_selection[np.isfinite(data_selection).any(axis=1)]

# target variable / covariates
y = data_selection[:,0:3]
x = data_selection[:,4:]

# split test-train
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size=0.2, random_state=0)


cca = CCA(n_components=1,scale=True)
cca.fit(x_train, y_train)
#CCA(copy=True, max_iter=500, n_components=1, scale=True, tol=1e-06),
X_train_r, Y_train_r = cca.transform(x_train,y_train)
X_test_r, Y_test_r = cca.transform(x_test, y_test)

print(type(X_train_r))
print(np.shape(X_train_r))
print(np.shape(Y_train_r))
print(np.shape(x))

print(np.corrcoef(X_train_r[:,0],Y_train_r[:,0]))
print(np.corrcoef(X_test_r[:,0],Y_test_r[:,0]))