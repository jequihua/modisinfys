
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import cross_validation
from sklearn import datasets
from sklearn.metrics import r2_score

# import training data as data frame
#data = pd.read_csv("D:/Julian/64_ie_maps/julian_tables_2/training_final_good.csv")
data = pd.read_csv("D:/Julian/64_ie_maps/cleaning_training/train_ff3.csv")

print(data.head())

# replace -999 flags
data = data.replace(-999,np.nan)


colnames = data.columns
print(colnames[284])

#print(np.shape(data))

data = data.loc\
		[\
		  (data.iloc[:,284] == 2004)\
		| (data.iloc[:,284] == 2005)\
		| (data.iloc[:,284] == 2006)\
		| (data.iloc[:,284] == 2007)\
		| (data.iloc[:,284] == 2009)\
		| (data.iloc[:,284] == 2010)\
		| (data.iloc[:,284] == 2011)\
		| (data.iloc[:,284] == 2012)\
		| (data.iloc[:,284] == 2013)\
		]

#print(np.shape(data))

# data structure
#print(data.head())

# initialize model
rf = RandomForestRegressor(n_estimators=3000,n_jobs=4,max_features=84,min_samples_split=5,oob_score=True)
#rf = ExtraTreesRegressor(n_estimators=500,n_jobs=4,max_features=100,min_samples_split=5)

# indices of variables of interest (target and covariates)
selection = np.append([270],range(4,257))
selection = np.append(selection,[284])

# select data of interest
data_selection = data.iloc[:,selection].as_matrix()

# check type of array
#print(np.dtype(data_selection))

# force dtype = float32
data_selection = data_selection.astype(np.float32, copy=False)

# complete cases
data_selection = data_selection[~np.isnan(data_selection).any(axis=1)]
data_selection = data_selection[np.isfinite(data_selection).any(axis=1)]

#np.savetxt("foo.csv",data_selection, delimiter=",")

# target variable / covariates
y = data_selection[:,0]
x = data_selection[:,1:]

# split test-train
#x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size=0.2, random_state=0)

# fit model
rfmodel = rf.fit(x,y)

corrins = np.corrcoef(rfmodel.oob_prediction_,y)
print(corrins)


# validation measures
#r2 = r2_score(y_test, rf.predict(x_test))
#mse = np.mean((y_test - rf.predict(x_test))**2)
#mae = np.mean(np.absolute((y_test - rf.predict(x_test))))

#print(r2)
#print(mse)
#print(mae)

# plot

# Print the feature ranking

# column names in searched_data_frame
colnames = data.columns


imp = rfmodel.feature_importances_
names = colnames[selection[1:]] 

imp,names = zip(*sorted(zip(imp,names)))
print(type(imp))
imp_selection = imp[0:21]
names_selection = names[0:21]

print(type(imp_selection))

plt.barh(range(21),imp_selection,align="center")
plt.yticks(range(21),names_selection)

plt.xlabel("Importance of features")
plt.ylabel("Features")
plt.title("Importance of eache feature")
plt.show()

# print("Feature ranking:")

# for f in range(10):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(10), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(10), indices)
# plt.xlim([-1, 10])
# plt.show()