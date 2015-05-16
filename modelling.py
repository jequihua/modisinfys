
import pandas as pd
import pylab as pl
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn import datasets
from sklearn.metrics import r2_score

# import training data as data frame
data = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/training_final.csv")
#data = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/datos_entrenamiento_altprom.csv")

# replace -999 flags
data = data.replace(-999,np.nan)

data = data.loc\
		[\
		  (data.iloc[:,26] == 2009)\
		| (data.iloc[:,26] == 2010)\
		| (data.iloc[:,26] == 2011)\
		| (data.iloc[:,26] == 2012)\
		| (data.iloc[:,26] == 2013)\
		]

# data structure
#print(data.head())

# initialize model
rf = RandomForestRegressor(n_estimators=1000,n_jobs=4,max_features=76,min_samples_split=5)

# indices of variables of interest (target and covariates)
selection = np.append([13],range(26,261))

# select data of interest
data_selection = data.iloc[:,selection].as_matrix()

# check type of array
#print(np.dtype(data_selection))

# force dtype = float32
data_selection = data_selection.astype(np.float32, copy=False)

# complete cases
data_selection = data_selection[~np.isnan(data_selection).any(axis=1)]
data_selection = data_selection[np.isfinite(data_selection).any(axis=1)]
data_selection[np.isinf(data_selection)]=0

np.savetxt("foo.csv",data_selection, delimiter=",")

# target variable / covariates
y = data_selection[:,0]
x = data_selection[:,1:]

# split test-train
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size=0.4, random_state=0)

# fit model
rfmodel = rf.fit(x_train,y_train)

# validation measures
r2 = r2_score(y_test, rf.predict(x_test))
mse = np.mean((y_test - rf.predict(x_test))**2)

print(r2)
print(mse)